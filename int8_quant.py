import torch
from torch import Tensor, nn
import torch.nn.functional as F

#The most important parts in the following are basically fully taken from OneTrainer.


# --- Triton Support ---
try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        ],
        key=['QUANTIZED_M', 'N', 'K', 'stride_bk'],
    )
    @triton.jit
    def __mm_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
            QUANTIZED_M,
            FLOAT: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32 if FLOAT else tl.int32)

        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
            b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32 if FLOAT else tl.int32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    def mm_8bit_triton(a: Tensor, b: Tensor) -> Tensor:
        FLOAT = (a.dtype == torch.float8_e4m3fn)
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float32 if FLOAT else torch.int32)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']))
        __mm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            QUANTIZED_M=M // 64, FLOAT=FLOAT
        )
        return c

except ImportError:
    mm_8bit_triton = None

def mm_8bit(a: Tensor, b: Tensor) -> Tensor:
    if mm_8bit_triton is not None and a.is_cuda:
        return mm_8bit_triton(a, b)
    
    if a.dtype == torch.int8:
        return torch._int_mm(a, b)
    else:
        one = torch.ones(1, device=a.device)
        return torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one)

# --- Quantization Utils ---

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale

def stochastic_round_int8_delta(x: Tensor, scale: float | Tensor, seed: int = 0) -> Tensor:
    """
    Quantize a delta tensor to INT8 using stochastic rounding.
    
    This is used for LoRA deltas to minimize quantization error while preserving
    the pre-quantized base weights' quality.
    
    Args:
        x: FP32 delta tensor to quantize
        scale: The quantization scale from the original INT8 weights
        seed: Random seed for reproducibility
    
    Returns:
        INT8 quantized delta tensor
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    
    # Scale to INT8 range (no zero-point for delta)
    x_scaled = x / scale
    
    # Stochastic rounding
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    
    # Clamp to INT8 range [-128, 127]
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# --- LinearW8A8 ---

@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

@torch.no_grad()
def int8_forward_static(x: Tensor, weight: Tensor, weight_scale: float | Tensor, input_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with static (learned) activation quantization."""
    # Quantize input using learned scale
    x_8 = quantize_int8(x, input_scale)
    res = torch._int_mm(x_8, weight.T)
    # Combined scale: weight_scale * input_scale
    res_scaled = res.float().mul_(weight_scale * input_scale).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

class LinearInt8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        return int8_forward_dynamic(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        raise NotImplementedError("Int W8A8 backward is not implemented for this ComfyUI node.")

class LinearW8A8(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("input_scale", None)  # Optional: learned input scale
        self.__is_quantized = False
        self.compute_dtype = torch.float16 # Default

    @torch.no_grad()
    def quantize(self):
        if self.__is_quantized:
            return
        self.requires_grad_(False)
        
        # Speed optimization: Perform quantization math on GPU if available
        # One layer at a time is very safe for VRAM.
        device = torch.device("cuda") if torch.cuda.is_available() else self.weight.device
        w_gpu = self.weight.data.to(device, non_blocking=True)
        
        weight, scale = quantize_int8_tensorwise(w_gpu)
        
        self.weight.data = weight.to(self.weight.device)
        self.scale.fill_(scale)
        self.__is_quantized = True

    @torch.no_grad()
    def load_quantized(self, weight, weight_scale, input_scale=None):
        self.requires_grad_(False)
        self.weight.data = weight.to(self.weight.device)
        self.scale.fill_(weight_scale)
        if input_scale is not None:
            if isinstance(input_scale, torch.Tensor):
                self.input_scale = input_scale.to(self.weight.device, dtype=torch.float32).view(())
            else:
                self.input_scale = torch.tensor(input_scale, dtype=torch.float32, device=self.weight.device)
        self.__is_quantized = True

    def forward(self, x_orig: Tensor) -> Tensor:
        if not self.__is_quantized:
            return super().forward(x_orig)
        
        x = x_orig.reshape(-1, x_orig.shape[-1])
        
        if x.shape[0] > 16:
            if self.input_scale is not None:
                # Use learned static quantization
                y = int8_forward_static(x, self.weight, self.scale, self.input_scale, self.bias, self.compute_dtype)
            else:
                # Fall back to dynamic quantization
                y = int8_forward_dynamic(x, self.weight, self.scale, self.bias, self.compute_dtype)
        else:
            # For small batches, dequantize and use fp math
            w = dequantize(self.weight, self.scale).to(x.dtype)
            y = F.linear(x, w, self.bias)
        
        return y.reshape(x_orig.shape[:-1] + (y.shape[-1],))

def apply_quantization(parent_module, compute_dtype=torch.float16, visited_modules=None, exclude_names=None, prefix="", state_dict=None):
    if visited_modules is None:
        visited_modules = set()
        # Track statistics for this run
        apply_quantization._stats = {"loaded_learned": 0, "quantized_on_fly": 0}
    if exclude_names is None:
        exclude_names = []
    
    if id(parent_module) in visited_modules:
        return parent_module
    visited_modules.add(id(parent_module))

    def should_exclude(name):
        return any(ex in name for ex in exclude_names)

    def replace_linear(module, full_name, parent, key):
        if should_exclude(full_name):
            return False
        
        new_module = LinearW8A8(module.in_features, module.out_features, module.bias is not None)
        
        # Check for pre-quantized weights in state_dict
        loaded_quant = False
        if state_dict is not None:
            weight_key = f"{full_name}.weight"
            scale_key = f"{full_name}.weight_scale"
            input_scale_key = f"{full_name}.input_scale"
            
            if weight_key in state_dict and scale_key in state_dict:
                try:
                    w_quant = state_dict[weight_key]
                    s_quant = state_dict[scale_key]
                    
                    # NOTE: input_scale is NOT loaded because it's typically a placeholder (1.0)
                    # from the conversion tool, not an actual learned value.
                    # The --heur option only learns weight rounding, not input scales.
                    # We use dynamic per-row activation quantization instead.
                    input_scale_val = None
                    
                    if w_quant.dtype == torch.int8:
                        # Handle tensor scale - extract scalar if needed
                        if isinstance(s_quant, torch.Tensor):
                            if s_quant.numel() == 1:
                                scale_val = s_quant.item()
                            else:
                                print(f"Warning: {full_name} has non-scalar weight_scale, re-quantizing")
                                scale_val = None
                        else:
                            scale_val = s_quant
                        
                        if scale_val is not None:
                            new_module.load_quantized(w_quant, scale_val, input_scale_val)
                            loaded_quant = True
                            apply_quantization._stats["loaded_learned"] += 1
                            if input_scale_val is not None:
                                apply_quantization._stats["with_input_scale"] = apply_quantization._stats.get("with_input_scale", 0) + 1
                except Exception as e:
                    print(f"Failed to load quantized weights for {full_name}: {e}")

        if not loaded_quant:
            new_module.weight.data.copy_(module.weight.data)
            new_module.quantize()
            apply_quantization._stats["quantized_on_fly"] += 1
        
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        
        new_module.compute_dtype = compute_dtype
        new_module.to(module.weight.device)
        
        if isinstance(key, int):
            parent[key] = new_module
        else:
            setattr(parent, key, new_module)
        return True

    if isinstance(parent_module, (nn.ModuleList, nn.Sequential)):
        for i, module in enumerate(parent_module):
            full_name = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(module, nn.Linear) and not isinstance(module, LinearW8A8):
                replace_linear(module, full_name, parent_module, i)
            else:
                apply_quantization(module, compute_dtype, visited_modules, exclude_names, full_name, state_dict)
    else:
        for name, module in parent_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Linear) and not isinstance(module, LinearW8A8):
                replace_linear(module, full_name, parent_module, name)
            else:
                apply_quantization(module, compute_dtype, visited_modules, exclude_names, full_name, state_dict)
    
    # Print stats at the end (when prefix is empty = top-level call)
    if not prefix:
        stats = apply_quantization._stats
        input_scale_count = stats.get("with_input_scale", 0)
        msg = f"Quantization complete: {stats['loaded_learned']} layers loaded from checkpoint"
        if input_scale_count > 0:
            msg += f" ({input_scale_count} with learned input_scale)"
        msg += f", {stats['quantized_on_fly']} layers quantized on-the-fly"
        print(msg)
    
    return parent_module


# =============================================================================
# INT8 LoRA Adapter - High Precision, Low RAM Patching
# =============================================================================

try:
    from comfy.weight_adapter.lora import LoRAAdapter
    _LORA_ADAPTER_AVAILABLE = True
except ImportError:
    _LORA_ADAPTER_AVAILABLE = False

if _LORA_ADAPTER_AVAILABLE:
    class INT8LoRAPatchAdapter(LoRAAdapter):
        """
        Specialized LoRA adapter that patches INT8 weights IN-PLACE in INT8 space.
        
        This avoids dequantizing the base model to FP32, saving GIGABYTES of RAM
        while maintaining perfect precision via stochastic rounding.
        """
        def __init__(self, loaded_keys, weights, weight_scale, seed=0):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            # weight: The base weight (ideally INT8 from convert_weight)
            # self.weights: (up, down, alpha, mid, dora, reshape)
            
            v = self.weights
            up, down, alpha = v[0], v[1], v[2]
            
            # 1. Resolve LoRA Scale
            rank = down.shape[0] if down.ndim >= 2 else 1
            scale = (alpha / rank) * strength if alpha is not None else strength
            
            device = weight.device
            
            # 2. Compute LoRA Delta in high-precision intermediate space
            # Use GPU for speed (one layer at a time is very safe for VRAM)
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
            up_f = up.to(comp_device, dtype=intermediate_dtype)
            down_f = down.to(comp_device, dtype=intermediate_dtype)
            
            # Handle possible mid weights (LoCon/LoHA)
            if v[3] is not None:
                mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
                # ... (simplified for now, full tucker expansion would go here)
                lora_diff = torch.mm(up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1))).reshape(weight.shape)
            else:
                lora_diff = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)
            
            # 3. Apply Patch
            if weight.dtype == torch.int8:
                # --- INT8 SPACE PATCHING (Giga-Brain Memory Efficiency) ---
                # We add the stochastically rounded delta directly to the bytes.
                delta_f = lora_diff * scale
                
                # stochastic_round_int8_delta is already GPU-optimized
                delta_int8 = stochastic_round_int8_delta(delta_f, self.weight_scale, self.seed)
                
                # Perform integer addition (int32 for safety)
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                # Fallback: Standard Float Patching (if weight was already dequantized by another patch)
                # Base_Float = Base_Int8 * Scale. So we add the float delta directly.
                return weight + (lora_diff * scale).to(weight.device, weight.dtype)

# =============================================================================
# Int8TensorwiseOps - Proper ComfyUI Custom Operations Class
# =============================================================================
# This replaces the old dequant→load→requant hack with direct int8 loading.

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """
        Custom ComfyUI operations for INT8 tensorwise quantization.
        
        This properly integrates with ComfyUI's model loading while keeping
        the blazing fast torch._int_mm forward path.
        """
        excluded_names = []
        
        class Linear(manual_cast.Linear):
            """Linear layer that directly loads int8 weights and uses fast _int_mm."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self.input_scale = None
                self._is_quantized = False
                self.compute_dtype = torch.bfloat16
            
            def reset_parameters(self):
                # Skip weight initialization - we load from state dict
                return None
            
            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """Directly load int8 weights and scales from state dict."""
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"
                
                weight_scale = state_dict.pop(scale_key, None)
                input_scale = state_dict.pop(input_scale_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)
                
                if weight_tensor is not None:
                    # Check if this is an int8 quantized weight
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        
                        # Store scale as scalar or tensor
                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self.weight_scale = weight_scale.float().item()
                            else:
                                self.weight_scale = weight_scale.float()
                        else:
                            self.weight_scale = float(weight_scale)
                        
                        # Store input scale if present
                        if input_scale is not None:
                            if isinstance(input_scale, torch.Tensor):
                                self.input_scale = input_scale.float()
                            else:
                                self.input_scale = torch.tensor(input_scale, dtype=torch.float32)
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # High-precision weight
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # Speed optimization: Quantize high-precision weights on GPU
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            w_gpu = weight_tensor.to(device, non_blocking=True)
                            q_weight, q_scale = quantize_int8_tensorwise(w_gpu)
                            
                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.weight_scale = q_scale.cpu() if isinstance(q_scale, torch.Tensor) else q_scale
                            self._is_quantized = True
                    else:
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)
                
                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

            def convert_weight(self, _weight, inplace=False):
                """
                Returns weight for patching.
                Returns raw INT8 to avoid FP32 model expansion in RAM.
                """
                if not self._is_quantized:
                    return _weight
                # We ignore the lossy cast usually passed in _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0):
                """Sets the patched weight."""
                if not self._is_quantized:
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight.to(self.weight.dtype), requires_grad=False)
                    return

                # If the patching happened in INT8 space, we just save it.
                if out_weight.dtype == torch.int8:
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # If patching resulted in floats (fallback), re-quantize.
                from .int8_quant import stochastic_round_int8_delta
                new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0):
                if out_bias is None: return
                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                """Fast forward using torch._int_mm for quantized weights."""
                
                if not self._is_quantized:
                    # Non-quantized path - use standard ComfyUI cast
                    weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out
                
                # Quantized path
                # NOTE: We intentionally do NOT use cast_bias_weight for the weight here.
                # cast_bias_weight in ComfyUI often forces a cast to BF16/FP16 which breaks _int_mm.
                # We manually move the weight to GPU while STRICTLY preserving Int8.
                
                # 1. Move weight to device (non_blocking for speed)
                weight = self.weight.to(x.device, non_blocking=True)
                
                # 2. Handle Bias (if present, usually needs to be moved)
                bias = None
                if self.bias is not None:
                    bias = self.bias.to(x.device, non_blocking=True)
                
                # 3. Handle Scales
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(x.device, non_blocking=True)
                
                
                i_scale = self.input_scale
                if i_scale is not None and isinstance(i_scale, torch.Tensor):
                    i_scale = i_scale.to(x.device, non_blocking=True)
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                # Flatten to 2D for matmul
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                
                # Use the appropriate forward based on batch size
                if x_2d.shape[0] > 16:
                    if i_scale is not None:
                        # Static quantization path
                        y = int8_forward_static(
                            x_2d, weight, w_scale,
                            i_scale, bias, compute_dtype
                        )
                    else:
                        # Dynamic activation quantization (default)
                        y = int8_forward_dynamic(
                            x_2d, weight, w_scale,
                            bias, compute_dtype
                        )
                else:
                    # Small batch - dequantize for accuracy
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    y = F.linear(x_2d, w_float, bias)
                
                # Reshape back
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        # Use standard ComfyUI implementations for non-Linear layers
        class GroupNorm(manual_cast.GroupNorm):
            pass
        
        class LayerNorm(manual_cast.LayerNorm):
            pass
        
        class Conv2d(manual_cast.Conv2d):
            pass
        
        class Conv3d(manual_cast.Conv3d):
            pass
        
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass
        
        class Embedding(manual_cast.Embedding):
            pass
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")
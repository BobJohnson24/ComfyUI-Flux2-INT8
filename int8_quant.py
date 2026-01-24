import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Add this at the top of your file
try:
    from .int8_fused_kernel import triton_int8_linear
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    print("Triton not found, falling back to torch._int_mm")

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
    Used for LoRA deltas to minimize quantization error.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    
    # Scale to INT8 range
    x_scaled = x / scale
    
    # Stochastic rounding
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# --- LinearW8A8 Core ---

@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    
    # --- FAST PATH: Triton Fused Kernel ---
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    
    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)
    
    # Dequantize: (res * weight_scale * x_scale)
    # Note: Creating intermediate Float tensors here is VRAM heavy
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled




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
        """
        def __init__(self, loaded_keys, weights, weight_scale, seed=0):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            v = self.weights
            up, down, alpha = v[0], v[1], v[2]
            
            rank = down.shape[0] if down.ndim >= 2 else 1
            scale = (alpha / rank) * strength if alpha is not None else strength
            
            device = weight.device
            
            # Compute LoRA Delta in high-precision on GPU
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
            up_f = up.to(comp_device, dtype=intermediate_dtype)
            down_f = down.to(comp_device, dtype=intermediate_dtype)
            
            # Handle possible mid weights (LoCon/LoHA)
            if v[3] is not None:
                mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
                lora_diff = torch.mm(up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1))).reshape(weight.shape)
            else:
                lora_diff = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)
            
            # Apply Patch
            if weight.dtype == torch.int8:
                # --- INT8 SPACE PATCHING ---
                delta_f = lora_diff * scale
                delta_int8 = stochastic_round_int8_delta(delta_f, self.weight_scale, self.seed)
                
                # Perform integer addition (int32 for safety) then clamp
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                # Fallback: Standard Float Patching
                return weight + (lora_diff * scale).to(weight.device, weight.dtype)


# =============================================================================
# Dynamic LoRA Synchronization Hook
# =============================================================================

class DynamicLoRAHook:
    """
    Hook registered on the diffusion_model to synchronize dynamic LoRA attributes
    with the current ModelPatcher context at the start of each forward pass.
    """
    def __init__(self):
        self.current_lora_id = None

    def pre_forward(self, module, input_args, input_kwargs):
        # 1. Try to find transformer_options
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            # Fallback for models that pass it in context
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]
        
        dynamic_loras = transformer_options.get("dynamic_loras", [])
        
        # 2. Generate a unique ID for this set of LoRAs
        # We use handles/strengths to detect changes
        lora_id = hash(tuple((id(d["patches"]), d["strength"]) for d in dynamic_loras)) if dynamic_loras else None
        
        if lora_id == self.current_lora_id:
            return None # Already synchronized
            
        # 3. Synchronize all linear layers
        self.apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    def apply_composition(self, diffusion_model, dynamic_loras):
        # Pre-group patches by layer
        layer_patches = {}
        if dynamic_loras:
            for entry in dynamic_loras:
                strength = entry["strength"]
                for key, adapter in entry["patches"].items():
                    if key not in layer_patches: layer_patches[key] = []
                    layer_patches[key].append((adapter, strength))

        # Update all modules
        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            
            # Find patches for this module
            # ComfyUI keys are often 'diffusion_model.path.to.weight' or 'path.to.weight'
            possible_keys = [f"diffusion_model.{name}.weight", f"{name}.weight"]
            patches = None
            for pk in possible_keys:
                if pk in layer_patches:
                    patches = layer_patches[pk]
                    break
            
            if not patches:
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            # Compose
            all_A = []
            all_B = []
            for adapter, strength in patches:
                v = adapter.weights
                up, down, alpha, mid = v[0], v[1], v[2], v[3]
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * strength if alpha is not None else strength
                
                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)
                
                all_A.append(curr_A * scale)
                all_B.append(up)
            
            if all_A:
                device = getattr(module, "weight", torch.tensor(0)).device
                module.lora_A = torch.cat(all_A, dim=0).to(device)
                module.lora_B = torch.cat(all_B, dim=1).to(device)
                module.lora_alpha = None
            else:
                module.lora_A = None
                module.lora_B = None

    @classmethod
    def register(cls, diffusion_model):
        if not hasattr(diffusion_model, "_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
        return diffusion_model._dynamic_lora_hook


# =============================================================================
# Int8TensorwiseOps - ComfyUI Custom Operations
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """
        Custom ComfyUI operations for INT8 tensorwise quantization.
        """
        excluded_names = []
        _is_prequantized = None # Global flag for current load
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self._is_quantized = False
                self.compute_dtype = torch.bfloat16
                self.lora_A = None
                self.lora_B = None
                self.lora_alpha = None
            
            def reset_parameters(self):
                return None
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"
                
                weight_scale = state_dict.pop(scale_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Load Quantized
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        Int8TensorwiseOps._is_prequantized = True # Found a quantized layer
                        
                        if isinstance(weight_scale, torch.Tensor):
                            self.weight_scale = weight_scale.float().item() if weight_scale.numel() == 1 else weight_scale.float()
                        else:
                            self.weight_scale = float(weight_scale)
                            
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # Load High-Precision
                        # Detect if the model is pre-quantized if we don't know yet
                        if Int8TensorwiseOps._is_prequantized is None:
                            # Robust detection: scan keys and a sample of values
                            is_prequant = False
                            for k in state_dict.keys():
                                if "weight_scale" in k or "comfy_quant" in k:
                                    is_prequant = True
                                    break
                            
                            if not is_prequant:
                                # Fallback: scan a sample of values for int8 tensors
                                for i, v in enumerate(state_dict.values()):
                                    if i > 200: break # Safety limit
                                    if getattr(v, "dtype", None) == torch.int8:
                                        is_prequant = True
                                        break
                            Int8TensorwiseOps._is_prequantized = is_prequant

                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1 or Int8TensorwiseOps._is_prequantized:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                            #print("Not quantizing", prefix)
                        else:
                            # Quantize on the fly
                            # We seriously need to avoid doing this when loading a prequantized model
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            w_gpu = weight_tensor.to(device, non_blocking=True)
                            q_weight, q_scale = quantize_int8_tensorwise(w_gpu)
                            #print("Quantizing", prefix)
                            
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
                if not self._is_quantized:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0):
                if not self._is_quantized:
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight.to(self.weight.dtype), requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # Re-quantize if fallback occurred
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
                    weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out
                
                # 1. Move weight/bias/scale to device (non_blocking)
                weight = self.weight.to(x.device, non_blocking=True)
                bias = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
                
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(x.device, non_blocking=True)
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                if x_2d.shape[0] > 16:
                    y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    y = F.linear(x_2d, w_float, bias)
                
                # Dynamic LoRA Path
                if self.lora_A is not None and self.lora_B is not None:
                    # Ensure LoRA tensors are on the same device as x
                    lA = self.lora_A.to(x.device, non_blocking=True)
                    lB = self.lora_B.to(x.device, non_blocking=True)
                    
                    lora_x = F.linear(x_2d.to(lA.dtype), lA)
                    lora_y = F.linear(lora_x, lB)
                    
                    if self.lora_alpha is not None:
                        lora_y = lora_y * self.lora_alpha
                    
                    y = y + lora_y.to(y.dtype)
                
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        # Pass-through for other layers
        class GroupNorm(manual_cast.GroupNorm): pass
        class LayerNorm(manual_cast.LayerNorm): pass
        class Conv2d(manual_cast.Conv2d): pass
        class Conv3d(manual_cast.Conv3d): pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding): pass
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2: return cls.Conv2d(*args, **kwargs)
            elif dims == 3: return cls.Conv3d(*args, **kwargs)
            else: raise ValueError(f"unsupported dimensions: {dims}")
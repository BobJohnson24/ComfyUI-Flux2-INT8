import torch
import os
from torch import Tensor, nn
import torch.nn.functional as F

# Add this at the top of your file
try:
    from .int8_fused_kernel import triton_int8_linear
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    print("Triton not found, falling back to torch._int_mm")

try:
    _disable_torch_compile = torch.compiler.disable
except Exception:
    def _disable_torch_compile(fn):
        return fn

try:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = max(0, int(os.environ.get("INT8_SMALL_BATCH_FALLBACK_MAX_ROWS", "16")))
except ValueError:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = 16

_DYNAMIC_LORA_DEBUG = os.environ.get("INT8_DYNAMIC_LORA_DEBUG", "0") == "1"

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
@_disable_torch_compile
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

@torch.no_grad()
@_disable_torch_compile
def apply_dynamic_lora_delta(
    x_2d: Tensor,
    y: Tensor,
    lora_A: Tensor | None,
    lora_B: Tensor | None,
    lora_alpha,
    lora_entries,
    device: torch.device,
) -> Tensor:
    if lora_entries:
        for entry in lora_entries:
            entry_A = entry.get("A")
            entry_B = entry.get("B")
            offset = entry.get("offset")
            if entry_A is None or entry_B is None:
                continue

            lA = entry_A if entry_A.device == device else entry_A.to(device, non_blocking=True)
            lB = entry_B if entry_B.device == device else entry_B.to(device, non_blocking=True)
            x_src = x_2d

            if offset is not None and len(offset) >= 3 and int(offset[0]) == 1:
                start = int(offset[1])
                length = int(offset[2])
                if start >= 0 and length > 0 and (start + length) <= x_2d.shape[-1]:
                    x_src = x_2d.narrow(-1, start, length)
                else:
                    if _DYNAMIC_LORA_DEBUG:
                        print(f"[INT8 Dynamic LoRA] skipping invalid input offset={offset} for x shape={tuple(x_2d.shape)}")
                    continue

            lora_x = F.linear(x_src.to(lA.dtype), lA)
            lora_y = F.linear(lora_x, lB).to(y.dtype)

            if offset is not None and len(offset) >= 3 and int(offset[0]) == 0:
                start = int(offset[1])
                length = int(offset[2])
                if lora_y.shape[-1] != length:
                    if _DYNAMIC_LORA_DEBUG:
                        print(
                            f"[INT8 Dynamic LoRA] skipping mismatched output slice "
                            f"offset={offset} lora_y={tuple(lora_y.shape)} y={tuple(y.shape)}"
                        )
                    continue
                if start >= 0 and length > 0 and (start + length) <= y.shape[-1]:
                    y.narrow(-1, start, length).add_(lora_y)
                else:
                    if _DYNAMIC_LORA_DEBUG:
                        print(f"[INT8 Dynamic LoRA] skipping invalid output offset={offset} for y shape={tuple(y.shape)}")
                continue

            if lora_y.shape[-1] != y.shape[-1]:
                if _DYNAMIC_LORA_DEBUG:
                    print(
                        f"[INT8 Dynamic LoRA] skipping mismatched full add "
                        f"lora_y={tuple(lora_y.shape)} y={tuple(y.shape)} offset={offset}"
                    )
                continue

            y.add_(lora_y)

        return y

    if lora_A is None or lora_B is None:
        return y

    lA = lora_A if lora_A.device == device else lora_A.to(device, non_blocking=True)
    lB = lora_B if lora_B.device == device else lora_B.to(device, non_blocking=True)

    lora_x = F.linear(x_2d.to(lA.dtype), lA)
    lora_y = F.linear(lora_x, lB)

    if lora_alpha is not None:
        lora_y = lora_y * lora_alpha

    return y + lora_y.to(y.dtype)




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

    class INT8MergedLoRAPatchAdapter(LoRAAdapter):
        """
        Adapter that merges multiple LoRAs in float space BEFORE applying a single
        stochastic rounding step. This is much more precise for LoRA stacks.
        """
        def __init__(self, patches, weight_scale, seed=0):
            # We need to satisfy the base LoRAAdapter constructor.
            # We use the first patch's keys/weights as a reference.
            first_patch_adapter = patches[0][0]
            super().__init__(first_patch_adapter.loaded_keys, first_patch_adapter.weights)
            
            # patches is a list of (adapter, strength)
            self.patches = patches
            self.weight_scale = weight_scale
            self.seed = seed

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            # Note: 'strength' from ComfyUI is ignored here as we use internal lora_strengths
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
            total_delta_f = None
            
            for adapter, lora_strength in self.patches:
                v = adapter.weights
                up, down, alpha = v[0], v[1], v[2]
                
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * lora_strength if alpha is not None else lora_strength
                
                up_f = up.to(comp_device, dtype=intermediate_dtype)
                down_f = down.to(comp_device, dtype=intermediate_dtype)
                
                if v[3] is not None:
                    mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
                    delta = torch.mm(up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1))).reshape(weight.shape)
                else:
                    delta = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)
                
                if total_delta_f is None:
                    total_delta_f = delta * scale
                else:
                    total_delta_f += delta * scale
            
            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                # One single stochastic rounding step for all combined LoRAs
                delta_int8 = stochastic_round_int8_delta(total_delta_f, self.weight_scale, self.seed)
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                return weight + total_delta_f.to(device, weight.dtype)


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

    @staticmethod
    def _compute_lora_id(dynamic_loras):
        if not dynamic_loras:
            return None
        # Use stable values so per-step dict/list cloning does not force recompose.
        return hash(tuple(
            (
                entry.get("name", ""),
                float(entry.get("strength", 0.0)),
                len(entry.get("patches", {})),
            )
            for entry in dynamic_loras
        ))

    @classmethod
    @_disable_torch_compile
    def sync_from_transformer_options(cls, diffusion_model, transformer_options):
        if transformer_options is None:
            transformer_options = {}

        dynamic_loras = transformer_options.get("dynamic_loras", [])
        target_modules = []

        if diffusion_model is not None:
            target_modules.append(diffusion_model)
            orig_mod = getattr(diffusion_model, "_orig_mod", None)
            if orig_mod is not None:
                target_modules.append(orig_mod)

        for module in target_modules:
            hook = cls.register(module)
            lora_id = cls._compute_lora_id(dynamic_loras)
            if lora_id == hook.current_lora_id:
                continue
            hook.apply_composition(module, dynamic_loras)
            hook.current_lora_id = lora_id

    @_disable_torch_compile
    def pre_forward(self, module, input_args, input_kwargs):
        # 1. Try to find transformer_options
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            # Fallback for models that pass it in context
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]
        
        dynamic_loras = transformer_options.get("dynamic_loras", [])
        
        # 2. Generate a stable ID for this set of LoRAs
        lora_id = self._compute_lora_id(dynamic_loras)
        
        if lora_id == self.current_lora_id:
            return None # Already synchronized
            
        # 3. Synchronize all linear layers
        self.apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    @_disable_torch_compile
    def apply_composition(self, diffusion_model, dynamic_loras):
        def normalize_patch_key(raw_key):
            key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
            if not isinstance(key, str):
                return None

            if key.endswith(".weight"):
                key = key[:-7]

            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model."):]
            elif key.startswith("model.diffusion_model."):
                key = key[len("model.diffusion_model."):]
            elif key.startswith("model."):
                key = key[len("model."):]

            if key.startswith("_orig_mod.diffusion_model."):
                key = key[len("_orig_mod.diffusion_model."):]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]

            return key

        def normalize_module_name(module_name):
            name = module_name
            if name.startswith("_orig_mod.diffusion_model."):
                name = name[len("_orig_mod.diffusion_model."):]
            elif name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
            elif name.startswith("diffusion_model."):
                name = name[len("diffusion_model."):]
            elif name.startswith("model.diffusion_model."):
                name = name[len("model.diffusion_model."):]
            elif name.startswith("model."):
                name = name[len("model."):]
            return name

        # Pre-group patches by layer
        layer_patches = {}
        if dynamic_loras:
            for entry in dynamic_loras:
                strength = entry["strength"]
                for key, adapter in entry["patches"].items():
                    normalized_key = normalize_patch_key(key)
                    if normalized_key is None:
                        continue
                    offset = key[1] if isinstance(key, tuple) and len(key) > 1 else None
                    if normalized_key not in layer_patches:
                        layer_patches[normalized_key] = []
                    layer_patches[normalized_key].append((adapter, strength, offset))

        # Update all modules
        candidate_modules = 0
        matched_modules = 0
        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            candidate_modules += 1
            
            normalized_name = normalize_module_name(name)
            patches = layer_patches.get(normalized_name)
            
            if not patches:
                module.dynamic_lora_entries = None
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            # Compose
            matched_modules += 1
            entries = []
            for adapter, strength, offset in patches:
                v = adapter.weights
                up, down, alpha, mid = v[0], v[1], v[2], v[3]
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * strength if alpha is not None else strength
                
                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)

                entries.append({
                    "A": curr_A * scale,
                    "B": up,
                    "offset": offset,
                })

            if entries:
                device = getattr(module, "weight", torch.tensor(0)).device
                module.dynamic_lora_entries = [
                    {
                        "A": entry["A"].to(device),
                        "B": entry["B"].to(device),
                        "offset": entry["offset"],
                    }
                    for entry in entries
                ]
            else:
                module.dynamic_lora_entries = None

            # Keep legacy fields unset to force offset-aware entry path.
            module.lora_A = None
            module.lora_B = None
            module.lora_alpha = None

        if _DYNAMIC_LORA_DEBUG:
            print(
                f"[INT8 Dynamic LoRA] candidate_modules={candidate_modules} "
                f"patch_keys={len(layer_patches)} matched_modules={matched_modules}"
            )

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
        dynamic_quantize = False # Manual toggle for on-the-fly quantization
        _is_prequantized = False # Keep this as a status flag, but don't use for detection
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self._is_quantized = False
                self.compute_dtype = torch.bfloat16
                self.dynamic_lora_entries = None
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
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # Quantize on the fly
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
                if not self._is_quantized:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight

                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight

                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # Re-quantize if fallback occurred
                new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
                
                if return_weight:
                    return new_weight

                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None: return None
                
                new_bias = out_bias
                if return_weight:
                    return new_bias

                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(new_bias)
                else:
                    self.bias = nn.Parameter(new_bias, requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                """Fast forward using torch._int_mm for quantized weights."""
                
                if not self._is_quantized:
                    weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out
                
                # 1. Move weight/bias/scale to device (non_blocking)
                weight = self.weight if self.weight.device == x.device else self.weight.to(x.device, non_blocking=True)
                if self.bias is None:
                    bias = None
                else:
                    bias = self.bias if self.bias.device == x.device else self.bias.to(x.device, non_blocking=True)
                
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    if w_scale.device != x.device:
                        w_scale = w_scale.to(x.device, non_blocking=True)
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                use_small_batch_fallback = _SMALL_BATCH_FALLBACK_MAX_ROWS > 0 and x_2d.shape[0] <= _SMALL_BATCH_FALLBACK_MAX_ROWS
                if use_small_batch_fallback:
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)
                else:
                    y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                
                # Dynamic LoRA Path
                y = apply_dynamic_lora_delta(
                    x_2d=x_2d,
                    y=y,
                    lora_A=self.lora_A,
                    lora_B=self.lora_B,
                    lora_alpha=self.lora_alpha,
                    lora_entries=self.dynamic_lora_entries,
                    device=x.device,
                )
                
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

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    
    # INT8 Matmul
    res = torch._int_mm(x_8, weight.T)
    
    # Dequantize: (res * weight_scale * x_scale)
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

class LinearW8A8(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.__is_quantized = False
        self.compute_dtype = torch.float16 # Default

    @torch.no_grad()
    def quantize(self):
        if self.__is_quantized:
            return
        self.requires_grad_(False)
        
        device = torch.device("cuda") if torch.cuda.is_available() else self.weight.device
        w_gpu = self.weight.data.to(device, non_blocking=True)
        
        weight, scale = quantize_int8_tensorwise(w_gpu)
        
        self.weight.data = weight.to(self.weight.device)
        self.scale.fill_(scale)
        self.__is_quantized = True

    @torch.no_grad()
    def load_quantized(self, weight, weight_scale, input_scale=None):
        """
        Load pre-quantized weights. 
        input_scale is accepted for compatibility but ignored.
        """
        self.requires_grad_(False)
        self.weight.data = weight.to(self.weight.device)
        self.scale.fill_(weight_scale)
        # input_scale is ignored; we strictly use dynamic quantization
        self.__is_quantized = True

    def forward(self, x_orig: Tensor) -> Tensor:
        if not self.__is_quantized:
            return super().forward(x_orig)
        
        x = x_orig.reshape(-1, x_orig.shape[-1])
        
        # Heuristic: For very small batches, dequantizing the weight 
        # is often faster/more accurate than quantizing the activation.
        if x.shape[0] > 16:
            y = int8_forward_dynamic(x, self.weight, self.scale, self.bias, self.compute_dtype)
        else:
            w = dequantize(self.weight, self.scale).to(x.dtype)
            y = F.linear(x, w, self.bias)
        
        return y.reshape(x_orig.shape[:-1] + (y.shape[-1],))

def apply_quantization(parent_module, compute_dtype=torch.float16, visited_modules=None, exclude_names=None, prefix="", state_dict=None):
    if visited_modules is None:
        visited_modules = set()
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
            
            if weight_key in state_dict and scale_key in state_dict:
                try:
                    w_quant = state_dict[weight_key]
                    s_quant = state_dict[scale_key]
                    
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
                            # input_scale (if exists in state_dict) is handled implicitly by not passing it,
                            # or strictly ignored by load_quantized.
                            new_module.load_quantized(w_quant, scale_val)
                            loaded_quant = True
                            apply_quantization._stats["loaded_learned"] += 1
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
    
    if not prefix:
        stats = apply_quantization._stats
        print(f"Quantization complete: {stats['loaded_learned']} layers loaded, {stats['quantized_on_fly']} quantized on-fly")
    
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
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self._is_quantized = False
                self.compute_dtype = torch.bfloat16
            
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
                        
                        if isinstance(weight_scale, torch.Tensor):
                            self.weight_scale = weight_scale.float().item() if weight_scale.numel() == 1 else weight_scale.float()
                        else:
                            self.weight_scale = float(weight_scale)
                            
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # Load High-Precision
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1:
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
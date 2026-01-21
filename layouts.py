# --- FILE: layouts.py ---
import torch
import torch.nn as nn
from comfy.quant_ops import QuantizedLayout

# Define the Custom Linear Layer that uses torch._int_mm
class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        # This is where the magic happens (torch._int_mm)
        # 1. Quantize Input X to int8
        input_scale = x.abs().max() / 127.0
        x_int8 = (x / input_scale).clamp(-127, 127).to(torch.int8)

        # 2. Perform Matrix Multiplication
        # Note: torch._int_mm requires inputs to be 2D, might need reshape handling for 3D tensors
        orig_shape = x_int8.shape
        out_int8 = torch._int_mm(
            x_int8.view(-1, x_int8.shape[-1]), 
            self.weight.t()
        )
        
        # 3. Dequantize Result
        out = out_int8.to(x.dtype) * (input_scale * self.weight_scale)
        
        # 4. Reshape and add Bias
        out = out.view(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out += self.bias
        return out

class Int8TensorwiseLayout(QuantizedLayout):
    """Layout that maps ComfyUI weights to our custom Int8Linear layer."""
    
    class Params:
        def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
            self.scale = scale
            self.orig_dtype = orig_dtype
            self.orig_shape = orig_shape
        
        def clone(self):
            return Int8TensorwiseLayout.Params(
                scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                orig_dtype=self.orig_dtype,
                orig_shape=self.orig_shape
            )

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # Defines how weights look in the safetensors file
        return {"": qdata, "weight_scale": params.scale}
    
    @classmethod  
    def dequantize(cls, qdata, params):
        # Fallback if the user wants to convert back to float
        return qdata.float() * params.scale

    def make_linear_op(self, device, dtype, bias=False, **kwargs):
        # THIS IS CRITICAL: It tells ComfyUI to use your custom layer
        # instead of a standard torch.nn.Linear
        return Int8Linear(self.orig_shape[1], self.orig_shape[0], bias=bias, device=device, dtype=dtype)
import torch
from comfy.quant_ops import QuantizedLayout

class Int8TensorwiseLayout(QuantizedLayout):
    """
    Minimal layout class for Int8 Tensorwise quantization.
    This tells ComfyUI how to read the weights from the file.
    """
    
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
        # Maps the data to the keys expected in the safetensors/checkpoint
        return {"": qdata, "weight_scale": params.scale}
    
    @classmethod  
    def dequantize(cls, qdata, params):
        # Helper for when Comfy needs to convert back to float (e.g. for VAES or non-supported ops)
        return qdata.float() * params.scale
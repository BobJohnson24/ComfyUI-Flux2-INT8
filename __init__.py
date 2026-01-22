"""
int88 - Fast INT8 Tensorwise Quantization for ComfyUI

Provides:
- Int8TensorwiseOps: Custom operations for direct int8 weight loading
- OTUNetLoaderW8A8: Load int8 quantized diffusion models
- OTCheckpointLoaderW8A8: Load int8 quantized checkpoints

Uses torch._int_mm for blazing fast inference.
"""

import logging
import torch

# =============================================================================
# Module-level state for comfy-kitchen backend integration
# =============================================================================

_CK_AVAILABLE = False
_CK_TRITON_AVAILABLE = False

def is_ck_triton_available() -> bool:
    """Check if comfy-kitchen triton backend is available and enabled."""
    return _CK_TRITON_AVAILABLE

# =============================================================================
# Backend Setup
# =============================================================================

def _setup_comfy_kitchen_backends():
    """
    Configure comfy-kitchen backends.
    1. Re-enable triton backend (ComfyUI disables it by default)
    2. Register our custom backend
    """
    global _CK_AVAILABLE, _CK_TRITON_AVAILABLE
    
    try:
        import comfy_kitchen as ck
        _CK_AVAILABLE = True
    except ImportError:
        # Fallback if comfy_kitchen isn't installed
        return
    
    # Step 1: Re-enable triton backend if available
    try:
        ck.enable_backend("triton")
        backends = ck.list_backends()
        triton_info = backends.get("triton", {})
        
        if triton_info.get("available") and not triton_info.get("disabled"):
            _CK_TRITON_AVAILABLE = True
        else:
            _CK_TRITON_AVAILABLE = False
            
    except Exception as e:
        logging.warning(f"Int88: Failed to enable ck triton backend: {e}")
        _CK_TRITON_AVAILABLE = False
    
    # Step 2: Register this node's kernels as a backend
    _register_quantops_backend()


def _register_quantops_backend():
    """
    Register kernels with comfy-kitchen registry.
    This allows the backend to dispatch to your optimized ops.
    """
    try:
        from comfy_kitchen.registry import registry
        from comfy_kitchen.constraints import (
            FunctionConstraints,
            ParamConstraint,
            ExactDims,
        )
        
        # Import your operations module
        # NOTE: Using .int8_quant based on your original file
        try:
            from . import int8_quant as ops_module
        except ImportError:
            logging.debug("Int88: Could not import int8_quant for backend registration")
            return

        cuda_devices = frozenset({"cuda"})
        standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
        
        # Define constraints for int8 tensorwise
        # This tells ComfyUI: "If inputs look like this, use my kernel"
        int8_constraints = {
            "linear": FunctionConstraints(
                params={
                    "input": ParamConstraint(dtypes=standard_floats),
                    "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                },
                default_devices=cuda_devices,
            ),
        }
        
        try:
            registry.register(
                name="int8_tensorwise_backend",
                module=ops_module,
                capabilities=int8_constraints,
                priority=10  # Priority over standard pytorch if matches
            )
            logging.info("Int88: Registered int8_tensorwise_backend")
        except Exception as e:
            logging.debug(f"Int88: Could not register backend: {e}")
            
    except ImportError:
        pass # comfy_kitchen not fully available or updated
    except Exception as e:
        logging.warning(f"Int88: Backend registration failed: {e}")


# =============================================================================
# Layout Registration
# =============================================================================

def _register_layouts():
    """
    Register the Int8Tensorwise layout with ComfyUI's model management.
    """
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout

        class Int8TensorwiseLayout(QuantizedLayout):
            """Minimal layout class to satisfy ComfyUI's registry requirements."""
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
                return {"": qdata, "weight_scale": params.scale}
            
            @classmethod  
            def dequantize(cls, qdata, params):
                return qdata.float() * params.scale

        # Register the class
        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        # Register the Algo Config
        # using setdefault ensures we don't overwrite if another node loaded it first,
        # but allows us to define the specs.
        QUANT_ALGOS.setdefault(
            "int8_tensorwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "Int8TensorwiseLayout",
                # tensorwise usually implies the whole tensor is one group, 
                # or typically handled implicitly by the kernel.
            }
        )
        
    except ImportError:
        logging.warning("Int88: ComfyUI Quantization system not found (Update ComfyUI?)")
    except Exception as e:
        logging.error(f"Int88: Failed to register layouts: {e}")

# =============================================================================
# Module Initialization
# =============================================================================

# 1. Setup Backends
_setup_comfy_kitchen_backends()

# 2. Register Layouts
_register_layouts()

# 3. Export Custom Ops (for external use)
try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

# 4. Node Mappings
from .int8_unet_loader import UNetLoaderINTW8A8
from .int8_lora import INT8LoraLoader

NODE_CLASS_MAPPINGS = {
    "OTUNetLoaderW8A8": UNetLoaderINTW8A8,
    "INT8LoraLoader": INT8LoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OTUNetLoaderW8A8": "Load Diffusion Model INT8 (W8A8)",
    "INT8LoraLoader": "Load LoRA INT8 (Stochastic)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "Int8TensorwiseOps",
]
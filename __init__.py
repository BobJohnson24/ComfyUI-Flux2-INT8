# --- FILE: __init__.py ---
import logging
import torch

# 1. Setup Internal Imports (Lazy loading prevents circular imports)
try:
    from .int8_unet_loader import UNetLoaderINTW8A8
    # If you have custom ops exposed for external use
    from .int8_quant import Int8TensorwiseOps 
except ImportError:
    UNetLoaderINTW8A8 = None
    Int8TensorwiseOps = None
    pass

# 2. Register Layouts (The "Proper" way)
def _register_layouts():
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class
        # Import the class from our separate file
        from .layouts import Int8TensorwiseLayout

        # Register the class string
        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        # Update QUANT_ALGOS
        # This tells ComfyUI: "When you see 'int8_tensorwise', use this layout"
        algo_config = {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
        }
        
        # Use setdefault to play nice with other nodes
        QUANT_ALGOS.setdefault("int8_tensorwise", algo_config)
        
        # Update existing if needed (in case ComfyUI loaded a default stub)
        if QUANT_ALGOS["int8_tensorwise"].get("comfy_tensor_layout") != "Int8TensorwiseLayout":
            QUANT_ALGOS["int8_tensorwise"].update(algo_config)

        logging.info("Int8-Quant: Registered Int8TensorwiseLayout successfully.")

    except ImportError:
        logging.warning("Int8-Quant: Failed to register layout. ComfyUI too old?")
    except Exception as e:
        logging.error(f"Int8-Quant: Error registering layout: {e}")

# Run registration immediately
_register_layouts()

# 3. Export Node Mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if UNetLoaderINTW8A8:
    NODE_CLASS_MAPPINGS["OTUNetLoaderW8A8"] = UNetLoaderINTW8A8
    NODE_DISPLAY_NAME_MAPPINGS["OTUNetLoaderW8A8"] = "Load Diffusion Model INT8 (W8A8)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
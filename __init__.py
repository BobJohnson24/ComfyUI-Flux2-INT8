import logging
import torch

# =============================================================================
# Layout Registration
# =============================================================================

def _register_layouts():
    """Register the INT8 layout and update ComfyUI's QUANT_ALGOS."""
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class
        # Import from our local file
        from .int8_layout import Int8TensorwiseLayout

        # 1. Register the class with ComfyUI
        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        # 2. Define the algorithm config
        algo_config = {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
            # Add specific flags if needed, e.g.:
            # "symmetric": True,
        }

        # 3. Add to QUANT_ALGOS (using setdefault to be safe)
        cur_config = QUANT_ALGOS.get("int8_tensorwise")
        if cur_config is None:
            QUANT_ALGOS["int8_tensorwise"] = algo_config
        else:
            # Update existing config if it exists (e.g. from another node)
            cur_config["comfy_tensor_layout"] = "Int8TensorwiseLayout"
            cur_config["parameters"] = {"weight_scale", "input_scale"}
            
        logging.info("Int8-Quant: Registered 'int8_tensorwise' layout successfully.")

    except ImportError:
        logging.warning("Int8-Quant: Could not import ComfyUI quant_ops. Update ComfyUI?")
    except Exception as e:
        logging.error(f"Int8-Quant: Failed to register layouts: {e}")

# Run registration immediately on import
_register_layouts()

# =============================================================================
# Node Exports
# =============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Lazy import nodes to avoid crashes if dependencies are missing
try:
    from .int8_unet_loader import UNetLoaderINTW8A8
    NODE_CLASS_MAPPINGS["OTUNetLoaderW8A8"] = UNetLoaderINTW8A8
    NODE_DISPLAY_NAME_MAPPINGS["OTUNetLoaderW8A8"] = "Load Diffusion Model INT8 (W8A8)"
except ImportError as e:
    logging.error(f"Int8-Quant: Could not import nodes: {e}")

# Optional: Export your custom Ops class if other nodes need it
try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
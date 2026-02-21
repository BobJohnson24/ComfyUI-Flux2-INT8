import folder_paths
import comfy.utils
import comfy.lora
import comfy.patcher_extension
import logging

_DYNAMIC_LORA_WRAPPER_KEY = "int8_dynamic_lora_sync"

def _dynamic_lora_sync_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    if transformer_options is None and len(args) > 5:
        transformer_options = args[5]
    if transformer_options is None:
        transformer_options = {}

    base_model = executor.class_obj
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is not None:
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.sync_from_transformer_options(diffusion_model, transformer_options)

    return executor(*args, **kwargs)

def _ensure_dynamic_sync_wrapper(model_patcher):
    model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, _DYNAMIC_LORA_WRAPPER_KEY)
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        _DYNAMIC_LORA_WRAPPER_KEY,
        _dynamic_lora_sync_wrapper
    )

class INT8DynamicLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()
        
        # 1. Get Patch Map
        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
        
        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        
        # 2. Register Global Hook (if not exists)
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        # 3. Add to Dynamic LoRA list in transformer_options
        # This ensures ComfyUI's cloning handles everything and it's non-sticky
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
        opts = model_patcher.model_options["transformer_options"]
        if "dynamic_loras" not in opts:
            opts["dynamic_loras"] = []
        else:
            # Shallow copy the list to avoid modifying the parent patcher's list
            opts["dynamic_loras"] = opts["dynamic_loras"].copy()
            
        opts["dynamic_loras"].append({
            "name": lora_name,
            "strength": strength,
            "patches": patch_dict
        })

        return (model_patcher,)

class INT8DynamicLoraStack:
    """
    Apply multiple LoRAs in one node for efficiency.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {"model": ("MODEL",)},
            "optional": {},
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "loaders"

    def apply_stack(self, model, **kwargs):
        lora_entries = []
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                lora_entries.append((lora_name, strength))

        if not lora_entries:
            return (model,)

        model_patcher = model.clone()

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}

        opts = model_patcher.model_options["transformer_options"]
        existing_loras = opts.get("dynamic_loras", [])
        opts["dynamic_loras"] = existing_loras.copy()

        for lora_name, strength in lora_entries:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patch_dict = comfy.lora.load_lora(lora_data, key_map, log_missing=True)

            opts["dynamic_loras"].append({
                "name": lora_name,
                "strength": strength,
                "patches": patch_dict
            })

        logging.info(f"INT8 Dynamic LoRA Stack: Loaded {len(lora_entries)} LoRAs in a single pass.")
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8DynamicLoraLoader": INT8DynamicLoraLoader,
    "INT8DynamicLoraStack": INT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8DynamicLoraLoader": "Load LoRA INT8 (Dynamic)",
    "INT8DynamicLoraStack": "INT8 LoRA Stack (Dynamic)",
}

import torch
import folder_paths
import comfy.utils
import comfy.lora
import logging

class INT8LoraLoader:
    """
    INT8 LoRA Loader that leverages ComfyUI's native patching system.
    
    Compatible with INT8 quantized layers via convert_weight and set_weight hooks 
    in int8_quant.py. Provides non-sticky, stackable, and RAM-efficient LoRA 
    application while preserving precision via stochastic rounding.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads LoRA for INT8 models with high-precision INT8-space patching. Avoids RAM bloat and preserves quality."

    def load_lora(self, model, lora_name, strength, seed):
        if strength == 0:
            return (model,)

        # Load LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Clone model patcher
        model_patcher = model.clone()
        
        # Get key mappings from ComfyUI's framework
        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
        
        # Use ComfyUI's load_lora to handle various LoRA formats
        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        
        # Upgrade patches for high-precision INT8-space patching
        from .int8_quant import INT8LoRAPatchAdapter
        
        final_patch_dict = {}
        applied_count = 0
        
        for key, adapter in patch_dict.items():
            # key can be "layer.name.weight" or ("layer.name", (dim, start, size))
            layer_name = key[0] if isinstance(key, tuple) else key
            if layer_name.endswith(".weight"):
                layer_name = layer_name[:-7]
            
            # Resolve module to check quantization status and get scale
            try:
                parts = layer_name.split(".")
                target_module = model_patcher.model.diffusion_model
                for part in parts[1:] if parts[0] == "diffusion_model" else parts:
                    if part.isdigit():
                        target_module = target_module[int(part)]
                    else:
                        target_module = getattr(target_module, part)
                
                # If module is quantized, upgrade the adapter to our high-precision version
                if hasattr(target_module, '_is_quantized') and target_module._is_quantized:
                    w_scale = target_module.weight_scale
                    if isinstance(w_scale, torch.Tensor):
                        w_scale = w_scale.item() if w_scale.numel() == 1 else w_scale
                    
                    # Create the specialized INT8 adapter
                    new_adapter = INT8LoRAPatchAdapter(
                        adapter.loaded_keys, 
                        adapter.weights, 
                        w_scale,
                        seed=seed
                    )
                    final_patch_dict[key] = new_adapter
                    applied_count += 1
                else:
                    final_patch_dict[key] = adapter
                    
            except (AttributeError, KeyError, IndexError, TypeError):
                final_patch_dict[key] = adapter

        # Add patches to the patcher. 
        # ComfyUI's ModelPatcher will apply our INT8LoRAPatchAdapter by:
        # 1. Calling module.convert_weight (returns raw INT8)
        # 2. Calling adapter.calculate_weight (does matmul and stochastic rounding in INT8 space)
        # 3. Calling module.set_weight (saves INT8 result)
        model_patcher.add_patches(final_patch_dict, strength)
        
        logging.info(f"INT8 LoRA: Registered '{lora_name}' with strength {strength:.2f} for {applied_count} quantized layers.")
        print(f"[INT8 LoRA] Patched {applied_count} layers, skipped {len(patch_dict) - applied_count}.")
        
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8LoraLoader": INT8LoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8LoraLoader": "Load LoRA INT8 (Stochastic)",
}
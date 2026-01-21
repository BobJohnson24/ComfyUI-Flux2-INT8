import os
import torch
import folder_paths
import comfy.sd
import comfy.utils

from .int8_quant import Int8TensorwiseOps


class UNetLoaderINTW8A8:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    Inference uses fast torch._int_mm for blazing speed. (insert rocket emoji, fire emoji to taste)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp16", "bf16"],),
                "model_type": (["flux2","z-image","chroma"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load INT8 tensorwise quantized models with fast torch._int_mm inference."

    def load_unet(self, unet_name, weight_dtype, model_type):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Use Int8TensorwiseOps for proper direct int8 loading
        model_options = {"custom_operations": Int8TensorwiseOps}
        
        # We need to peek at the model type to set exclusions for Flux
        # ComfyUI loads metadata before the full model
        from comfy.sd import load_diffusion_model
        
        # Reset exclusions (in case this is the second load)
        Int8TensorwiseOps.excluded_names = []
        
        # Check explicit model_type for exclusions
        if model_type == "flux2":
            Int8TensorwiseOps.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 'final_layer', 
                'double_stream_modulation_img', 'double_stream_modulation_txt', 
                'single_stream_modulation'
            ]
        elif model_type == "z-image":
            Int8TensorwiseOps.excluded_names = [
                'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner', 
                'final_layer', 'noise_refiner', 
                'x_pad_token'
            ]
        elif model_type == "chroma":
            Int8TensorwiseOps.excluded_names = [
                'distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in' 'nerf_image_embedder',
                 'nerf_blocks', 'nerf_final_layer_conv'
            ]
            #print(f"Applying Flux2-specific exclusions to Int8TensorwiseOps: {Int8TensorwiseOps.excluded_names}")

        # Load model directly - Int8TensorwiseOps handles int8 weights natively
        model = load_diffusion_model(unet_path, model_options=model_options)
        
        return (model,)


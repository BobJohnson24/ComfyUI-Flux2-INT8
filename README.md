# Flux2 INT8 Acceleration

This node speeds up Flux2 in ComfyUI by using INT8 quantization, delivering ~2x faster inference on my 3090, but it should work on any NVIDIA GPU with enough INT8 TOPS. It's unlikely to be faster than proper FP8 on 40-Series and above. 
Works with lora*, torch compile (needed to get full speedup).

*LoRAs need to be applied using one of the following methods:

### Option 1: Included INT8 LoRA Node (Recommended for Speed)
- **Performance:** Faster inference
- **Quality:** Possibly slightly lower quality
- Use the included INT8 LoRA node

### Option 2: KohakuBlueleaf's Node
- **Performance:** ~1.15x slower due to dynamic calculations
- **Quality:** Possibly slightly higher quality
- Requires the node from [KohakuBlueleaf's PR #11958](https://github.com/Comfy-Org/ComfyUI/pull/11958)

We auto-convert flux2 klein to INT8 on load if needed. Pre-quantized checkpoints with slightly higher quality and enabling faster loading are available here: 
https://huggingface.co/bertbobson/FLUX.2-klein-9B-INT8-Comfy

# Metrics:

Measured at 1024x1024, 26 steps with Flux2 Klein Base 9B.

| Format | Speed (s/it) | Relative Speedup |
|-------|--------------|------------------|
| bf16 | 2.07 | 1.00× |
| bf16 compile | 2.24 | 0.92× |
| fp8 | 2.06 | 1.00× |
| int8 | 1.64 | 1.26× |
| int8 compile | 1.04 | 1.99× |
| gguf8_0 compile | 2.03 | 1.02× |



# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!


## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of a day

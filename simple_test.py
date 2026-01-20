#!/usr/bin/env python3
"""
Minimal example for quick I2V + LoRA testing.
Edit the variables below and run: python simple_test.py
"""

from PIL import Image
from wan.image2video import WanI2V
from wan.configs.wan_i2v_A14B import i2v_A14B as config
from wan.utils.utils import save_video

# ============================================================================
# EDIT THESE VARIABLES
# ============================================================================

# Required paths
CHECKPOINT_DIR = "./Wan2.2-I2V-A14B"
INPUT_IMAGE = "./examples/i2v_input.JPG"
OUTPUT_VIDEO = "./output_test.mp4"

# LoRA paths (set to None to disable)
LORA_PATHS = [
    # "./loras/style.safetensors",
]
LORA_SCALES = [1.0]  # One scale per LoRA

# Expert-specific LoRAs (set to None to disable)
HIGH_NOISE_LORA = None  # "./loras/high_noise.safetensors"
LOW_NOISE_LORA = None   # "./loras/low_noise.safetensors"

# Generation parameters
PROMPT = "A cinematic video with smooth camera motion"
RESOLUTION = "1280*720"  # Options: "480*832", "720*1280", "1280*720"
FRAMES = 81              # Must be 4n+1 (e.g., 81, 161)
STEPS = 40               # Higher = better quality but slower
CFG_SCALE = 5.0          # Guidance scale (1.0-15.0)
SEED = 42                # -1 for random

# Memory settings
OFFLOAD_MODEL = True     # Reduces VRAM usage
USE_FP16 = True          # Reduces memory
T5_ON_CPU = False        # Move T5 to CPU if OOM

# Debug
VERBOSE = True           # Show detailed LoRA logs

# ============================================================================
# RUN INFERENCE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Wan2.2 I2V + LoRA Simple Test")
    print("=" * 80)

    # Load image
    print(f"Loading image: {INPUT_IMAGE}")
    img = Image.open(INPUT_IMAGE)
    print(f"Image size: {img.size}")

    # Calculate max area
    width, height = map(int, RESOLUTION.split('*'))
    max_area = width * height

    # Handle LoRA paths
    lora_paths = LORA_PATHS if LORA_PATHS else None
    lora_scales = LORA_SCALES if lora_paths else None
    high_lora_paths = [HIGH_NOISE_LORA] if HIGH_NOISE_LORA else None
    low_lora_paths = [LOW_NOISE_LORA] if LOW_NOISE_LORA else None

    # Print LoRA config
    if lora_paths:
        print("\nLoRA Configuration (both experts):")
        for i, lora in enumerate(lora_paths):
            scale = lora_scales[i] if lora_scales and i < len(lora_scales) else 1.0
            print(f"  - {lora} (scale={scale})")

    if high_lora_paths and high_lora_paths[0]:
        print(f"\nHigh-Noise Expert LoRA: {high_lora_paths[0]}")

    if low_lora_paths and low_lora_paths[0]:
        print(f"Low-Noise Expert LoRA: {low_lora_paths[0]}")

    print(f"\nPrompt: {PROMPT}")
    print(f"Resolution: {RESOLUTION}")
    print(f"Frames: {FRAMES}, Steps: {STEPS}, CFG: {CFG_SCALE}, Seed: {SEED}")
    print("=" * 80)

    # Initialize pipeline
    print("\n[1/3] Initializing pipeline (may take 1-2 minutes)...")
    pipe = WanI2V(
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        device_id=0,
        convert_model_dtype=USE_FP16,
        t5_cpu=T5_ON_CPU,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        high_noise_lora_paths=high_lora_paths,
        low_noise_lora_paths=low_lora_paths,
        lora_verbose=VERBOSE,
    )

    print("✓ Pipeline ready!")

    # Generate video
    print("\n[2/3] Generating video (this may take 5-15 minutes)...")
    video = pipe.generate(
        input_prompt=PROMPT,
        img=img,
        max_area=max_area,
        frame_num=FRAMES,
        sampling_steps=STEPS,
        guide_scale=CFG_SCALE,
        seed=SEED,
        offload_model=OFFLOAD_MODEL,
    )

    print("✓ Video generated!")

    # Save
    print(f"\n[3/3] Saving to: {OUTPUT_VIDEO}")
    save_video(
        tensor=video[None],
        save_file=OUTPUT_VIDEO,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    print("=" * 80)
    print("✓ DONE!")
    print(f"Output: {OUTPUT_VIDEO}")
    print("=" * 80)

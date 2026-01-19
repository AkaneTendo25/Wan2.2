"""
Example script demonstrating Image-to-Video generation with LoRA support in Wan2.2.

This script shows three different ways to use LoRA with the Wan2.2-I2V-A14B model:
1. Applying the same LoRA to both high and low noise models
2. Using separate LoRAs for high and low noise models
3. Using multiple LoRAs with different scaling factors
"""

import os
import sys
from PIL import Image
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.image2video import WanI2V
from wan.configs.wan_i2v_A14B import i2v_A14B as config


def example1_shared_lora():
    """Example 1: Apply the same LoRA to both high and low noise models."""
    print("=" * 80)
    print("Example 1: Shared LoRA for both high and low noise models")
    print("=" * 80)

    checkpoint_dir = "./Wan2.2-I2V-A14B"
    lora_path = "./loras/style_lora.safetensors"

    # Initialize pipeline with shared LoRA
    pipe = WanI2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        offload_model=True,
        convert_model_dtype=True,
        # Apply same LoRA to both models
        lora_paths=[lora_path],
        lora_scales=[1.0],  # Full strength
        lora_verbose=True,  # Show detailed logs
    )

    # Load input image
    img = Image.open("examples/i2v_input.JPG")
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression."
    )

    # Generate video
    video = pipe.generate(
        input_prompt=prompt,
        img=img,
        max_area=1280 * 720,
        frame_num=81,
        seed=42,
    )

    print(f"Generated video shape: {video.shape}")
    # Save video...


def example2_separate_loras():
    """Example 2: Use separate LoRAs for high and low noise models."""
    print("=" * 80)
    print("Example 2: Separate LoRAs for high and low noise models")
    print("=" * 80)

    checkpoint_dir = "./Wan2.2-I2V-A14B"
    high_noise_lora = "./loras/high_noise_lora.safetensors"
    low_noise_lora = "./loras/low_noise_lora.safetensors"

    # Initialize pipeline with separate LoRAs
    pipe = WanI2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        offload_model=True,
        convert_model_dtype=True,
        # Apply different LoRAs to each model
        high_noise_lora_paths=[high_noise_lora],
        high_noise_lora_scales=[1.0],
        low_noise_lora_paths=[low_noise_lora],
        low_noise_lora_scales=[1.0],
        lora_verbose=True,
    )

    # Load input image
    img = Image.open("examples/i2v_input.JPG")
    prompt = "A cinematic shot with dramatic lighting and composition"

    # Generate video
    video = pipe.generate(
        input_prompt=prompt,
        img=img,
        max_area=1280 * 720,
        frame_num=81,
        seed=42,
    )

    print(f"Generated video shape: {video.shape}")
    # Save video...


def example3_multiple_loras():
    """Example 3: Use multiple LoRAs with different scaling factors."""
    print("=" * 80)
    print("Example 3: Multiple LoRAs with custom scaling")
    print("=" * 80)

    checkpoint_dir = "./Wan2.2-I2V-A14B"

    # Multiple LoRAs for different effects
    lora_paths = [
        "./loras/style_lora.safetensors",
        "./loras/motion_lora.safetensors",
        "./loras/quality_lora.safetensors",
    ]

    # Custom scaling: style at 80%, motion at 100%, quality at 50%
    lora_scales = [0.8, 1.0, 0.5]

    # Initialize pipeline with multiple LoRAs
    pipe = WanI2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        offload_model=True,
        convert_model_dtype=True,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        lora_verbose=True,
    )

    # Load input image
    img = Image.open("examples/i2v_input.JPG")
    prompt = "Smooth camera movement with artistic style and high quality details"

    # Generate video
    video = pipe.generate(
        input_prompt=prompt,
        img=img,
        max_area=1280 * 720,
        frame_num=81,
        seed=42,
    )

    print(f"Generated video shape: {video.shape}")
    # Save video...


def example4_expert_specific_loras():
    """Example 4: Apply multiple LoRAs per expert with custom scaling."""
    print("=" * 80)
    print("Example 4: Multiple LoRAs per expert model")
    print("=" * 80)

    checkpoint_dir = "./Wan2.2-I2V-A14B"

    # High noise expert: focus on composition and layout
    high_noise_loras = [
        "./loras/composition_lora.safetensors",
        "./loras/layout_lora.safetensors",
    ]
    high_noise_scales = [1.0, 0.7]

    # Low noise expert: focus on details and quality
    low_noise_loras = [
        "./loras/detail_lora.safetensors",
        "./loras/quality_lora.safetensors",
    ]
    low_noise_scales = [1.0, 0.5]

    # Initialize pipeline
    pipe = WanI2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        offload_model=True,
        convert_model_dtype=True,
        high_noise_lora_paths=high_noise_loras,
        high_noise_lora_scales=high_noise_scales,
        low_noise_lora_paths=low_noise_loras,
        low_noise_lora_scales=low_noise_scales,
        lora_verbose=True,
    )

    # Load input image
    img = Image.open("examples/i2v_input.JPG")
    prompt = "Professional cinematography with detailed textures and refined quality"

    # Generate video
    video = pipe.generate(
        input_prompt=prompt,
        img=img,
        max_area=1280 * 720,
        frame_num=81,
        seed=42,
    )

    print(f"Generated video shape: {video.shape}")
    # Save video...


if __name__ == "__main__":
    # Run example 1 (modify as needed)
    example1_shared_lora()

    # Uncomment to run other examples:
    # example2_separate_loras()
    # example3_multiple_loras()
    # example4_expert_specific_loras()

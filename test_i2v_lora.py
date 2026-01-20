#!/usr/bin/env python3
"""
Standalone script for testing Wan2.2 I2V inference with LoRA.
Simple interface for quick testing and parameter tuning.

Usage:
    python test_i2v_lora.py \
        --checkpoint ./Wan2.2-I2V-A14B \
        --image ./input.jpg \
        --lora ./loras/style.safetensors \
        --output ./output.mp4

    # With expert-specific LoRAs:
    python test_i2v_lora.py \
        --checkpoint ./Wan2.2-I2V-A14B \
        --image ./input.jpg \
        --high_noise_lora ./loras/composition.safetensors \
        --low_noise_lora ./loras/details.safetensors \
        --output ./output.mp4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wan.image2video import WanI2V
from wan.configs.wan_i2v_A14B import i2v_A14B as config
from wan.utils.utils import save_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test I2V inference with LoRA"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Wan2.2-I2V-A14B checkpoint directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora",
        type=str,
        action="append",
        default=None,
        help="Path to LoRA .safetensors file (can specify multiple times)"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        action="append",
        default=None,
        help="Scale for each LoRA (default: 1.0)"
    )
    parser.add_argument(
        "--high_noise_lora",
        type=str,
        action="append",
        default=None,
        help="LoRA for high-noise expert only"
    )
    parser.add_argument(
        "--high_noise_lora_scale",
        type=float,
        action="append",
        default=None,
        help="Scale for high-noise LoRA"
    )
    parser.add_argument(
        "--low_noise_lora",
        type=str,
        action="append",
        default=None,
        help="LoRA for low-noise expert only"
    )
    parser.add_argument(
        "--low_noise_lora_scale",
        type=float,
        action="append",
        default=None,
        help="Scale for low-noise LoRA"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic video with smooth motion",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280*720",
        choices=["480*832", "720*1280", "1280*720"],
        help="Output resolution (area)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames (must be 4n+1, e.g., 81, 161)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Sampling steps (higher = better quality, slower)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)"
    )

    # Memory optimization
    parser.add_argument(
        "--offload",
        action="store_true",
        default=True,
        help="Offload models to CPU (reduces VRAM)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16/BF16 for model weights"
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Keep T5 encoder on CPU"
    )

    # Debug
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable detailed LoRA logs"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID"
    )

    return parser.parse_args()


def validate_args(args):
    """Validate arguments."""
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Check image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Check LoRA files exist
    for lora_path in (args.lora or []):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA not found: {lora_path}")

    for lora_path in (args.high_noise_lora or []):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"High-noise LoRA not found: {lora_path}")

    for lora_path in (args.low_noise_lora or []):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Low-noise LoRA not found: {lora_path}")

    # Validate frames
    if (args.frames - 1) % 4 != 0:
        raise ValueError(f"Frames must be 4n+1 (e.g., 81, 161), got {args.frames}")

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")


def print_config(args):
    """Print configuration."""
    logger.info("=" * 80)
    logger.info("Wan2.2 I2V Inference with LoRA")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Resolution: {args.resolution}")
    logger.info(f"Frames: {args.frames}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"CFG Scale: {args.cfg_scale}")
    logger.info(f"Seed: {args.seed if args.seed >= 0 else 'random'}")

    if args.lora:
        logger.info("\nLoRA Configuration (both experts):")
        for i, lora in enumerate(args.lora):
            scale = args.lora_scale[i] if args.lora_scale and i < len(args.lora_scale) else 1.0
            logger.info(f"  {i+1}. {lora} (scale={scale})")

    if args.high_noise_lora:
        logger.info("\nHigh-Noise Expert LoRAs:")
        for i, lora in enumerate(args.high_noise_lora):
            scale = args.high_noise_lora_scale[i] if args.high_noise_lora_scale and i < len(args.high_noise_lora_scale) else 1.0
            logger.info(f"  {i+1}. {lora} (scale={scale})")

    if args.low_noise_lora:
        logger.info("\nLow-Noise Expert LoRAs:")
        for i, lora in enumerate(args.low_noise_lora):
            scale = args.low_noise_lora_scale[i] if args.low_noise_lora_scale and i < len(args.low_noise_lora_scale) else 1.0
            logger.info(f"  {i+1}. {lora} (scale={scale})")

    logger.info("=" * 80)


def main():
    args = parse_args()

    # Validate
    validate_args(args)

    # Print config
    print_config(args)

    # Load image
    logger.info("Loading input image...")
    img = Image.open(args.image)
    logger.info(f"Image size: {img.size}")

    # Parse resolution
    width, height = map(int, args.resolution.split('*'))
    max_area = width * height

    # Initialize pipeline
    logger.info("Initializing WanI2V pipeline...")
    logger.info("(This may take a few minutes on first load...)")

    pipe = WanI2V(
        config=config,
        checkpoint_dir=args.checkpoint,
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.fp16,
        # LoRA parameters
        lora_paths=args.lora,
        lora_scales=args.lora_scale,
        high_noise_lora_paths=args.high_noise_lora,
        high_noise_lora_scales=args.high_noise_lora_scale,
        low_noise_lora_paths=args.low_noise_lora,
        low_noise_lora_scales=args.low_noise_lora_scale,
        lora_verbose=args.verbose,
    )

    logger.info("Pipeline initialized successfully!")

    # Generate video
    logger.info("Generating video...")
    logger.info(f"This will take several minutes depending on your GPU...")

    video = pipe.generate(
        input_prompt=args.prompt,
        img=img,
        max_area=max_area,
        frame_num=args.frames,
        shift=5.0 if max_area >= 1280*720 else 3.0,
        sample_solver='unipc',
        sampling_steps=args.steps,
        guide_scale=args.cfg_scale,
        seed=args.seed,
        offload_model=args.offload,
    )

    # Save video
    logger.info(f"Saving video to: {args.output}")
    save_video(
        tensor=video[None],
        save_file=args.output,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    logger.info("=" * 80)
    logger.info("✓ Done!")
    logger.info(f"Output saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

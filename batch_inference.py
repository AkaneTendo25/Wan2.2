#!/usr/bin/env python3
"""
Batch LoRA Testing Framework for Wan2.2 I2V

Automatically discovers LoRAs, runs inference with multiple configurations,
and saves results for A/B testing. Supports resume on failure.

Usage:
    python batch_inference.py --config batch_config.yaml
    python batch_inference.py --config batch_config.yaml --dry-run
    python batch_inference.py --config batch_config.yaml --resume
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wan.image2video import WanI2V
from wan.configs.wan_i2v_A14B import i2v_A14B as config
from wan.utils.utils import save_video


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LoRAInfo:
    """Information about a discovered LoRA."""
    path: str
    name: str  # Unique name derived from path
    relative_path: str  # Relative to lora_root


@dataclass
class TestCase:
    """A single test case configuration."""
    lora: Optional[LoRAInfo]  # None for baseline
    test_name: str
    image_path: Optional[str]  # None for T2V
    prompt: str
    seed: int
    scale: float
    output_path: str
    metadata_path: str
    is_baseline: bool = False
    is_t2v: bool = False


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_case: TestCase
    success: bool
    error: Optional[str]
    generation_time: float
    gpu_memory_peak: float
    timestamp: str


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class StateManager:
    """Manages execution state for resume capability."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self.load()

    def load(self) -> Dict:
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load state file: {e}")
        return {
            'completed': [],
            'failed': [],
            'start_time': None,
            'last_update': None,
        }

    def save(self):
        """Save state to file."""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_completed(self, test_id: str) -> bool:
        """Check if test is already completed."""
        return test_id in self.state['completed']

    def mark_completed(self, test_id: str):
        """Mark test as completed."""
        if test_id not in self.state['completed']:
            self.state['completed'].append(test_id)
        self.save()

    def mark_failed(self, test_id: str, error: str):
        """Mark test as failed."""
        self.state['failed'].append({
            'test_id': test_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'completed': len(self.state['completed']),
            'failed': len(self.state['failed']),
            'start_time': self.state.get('start_time'),
            'last_update': self.state.get('last_update'),
        }


# ============================================================================
# LORA DISCOVERY
# ============================================================================

def discover_loras(
    lora_root: str,
    lora_filename: str,
    max_depth: int,
    exclude_patterns: List[str],
    filter_subdirs: Optional[List[str]]
) -> List[LoRAInfo]:
    """
    Discover all LoRA files in directory tree.

    Args:
        lora_root: Root directory to search
        lora_filename: Filename to look for (e.g., "adapter_model.safetensors")
        max_depth: Maximum directory depth to search
        exclude_patterns: Patterns to exclude
        filter_subdirs: Only include specific subdirectories (None = all)

    Returns:
        List of discovered LoRA files
    """
    loras = []
    lora_root_path = Path(lora_root).resolve()

    logging.info(f"Discovering LoRAs in: {lora_root}")
    logging.info(f"Looking for files named: {lora_filename}")

    for root, dirs, files in os.walk(lora_root):
        # Calculate depth
        depth = len(Path(root).relative_to(lora_root_path).parts)
        if depth > max_depth:
            dirs.clear()  # Don't recurse deeper
            continue

        # Apply exclusions
        dirs[:] = [d for d in dirs if not any(
            Path(os.path.join(root, d)).match(pattern)
            for pattern in exclude_patterns
        )]

        # Check for LoRA file
        if lora_filename in files:
            lora_path = os.path.join(root, lora_filename)
            relative_path = os.path.relpath(lora_path, lora_root)

            # Apply filter
            if filter_subdirs:
                if not any(subdir in relative_path for subdir in filter_subdirs):
                    continue

            # Generate unique name from path
            name = Path(lora_path).parent.name + "_" + hashlib.md5(
                relative_path.encode()
            ).hexdigest()[:8]

            loras.append(LoRAInfo(
                path=lora_path,
                name=name,
                relative_path=relative_path
            ))

    logging.info(f"Discovered {len(loras)} LoRA files")
    return loras


# ============================================================================
# TEST CASE GENERATION
# ============================================================================

def generate_test_cases(
    loras: List[LoRAInfo],
    cfg: Dict,
    output_dir: str
) -> List[TestCase]:
    """Generate all test cases from configuration."""
    test_cases = []

    tests = cfg['test_inputs']['tests']
    seeds = cfg['generation']['seeds']

    # Handle scales
    test_scales = cfg['lora'].get('test_scales')
    if test_scales:
        scales = test_scales
    else:
        scales = [cfg['lora']['default_scale']]

    # Generate baseline cases first if enabled
    if cfg['advanced']['include_baseline']:
        logging.info("Generating baseline test cases (no LoRA)...")
        baseline_output_dir = os.path.join(
            output_dir,
            cfg['advanced']['baseline_output_dir']
        )
        for test in tests:
            for seed in seeds:
                test_cases.append(_create_test_case(
                    lora=None,  # No LoRA
                    test_config=test,
                    seed=seed,
                    scale=1.0,  # Not used for baseline
                    cfg=cfg,
                    output_dir=baseline_output_dir,
                    is_baseline=True
                ))

    # Generate LoRA test cases
    for lora in loras:
        for test in tests:
            for seed in seeds:
                for scale in scales:
                    test_cases.append(_create_test_case(
                        lora=lora,
                        test_config=test,
                        seed=seed,
                        scale=scale,
                        cfg=cfg,
                        output_dir=output_dir,
                        is_baseline=False
                    ))

    logging.info(f"Generated {len(test_cases)} test cases")
    return test_cases


def _create_test_case(
    lora: Optional[LoRAInfo],
    test_config: Dict,
    seed: int,
    scale: float,
    cfg: Dict,
    output_dir: str,
    is_baseline: bool = False
) -> TestCase:
    """Create a single test case."""
    # Determine output path based on structure
    structure = cfg['output']['structure']
    filename_template = cfg['output']['filename_template']

    test_name = test_config['name']
    image_path = test_config.get('image')
    prompt = test_config['prompt']
    is_t2v = image_path is None

    # Determine base directory based on structure
    if structure == "flat":
        base_dir = output_dir
    elif structure == "by_lora":
        lora_name = "baseline" if is_baseline else lora.name
        base_dir = os.path.join(output_dir, lora_name)
    elif structure == "by_test":
        base_dir = os.path.join(output_dir, test_name)
    elif structure == "hierarchical":
        lora_name = "baseline" if is_baseline else lora.name
        base_dir = os.path.join(output_dir, lora_name, test_name)
    elif structure == "preserve":
        if is_baseline:
            base_dir = output_dir
        else:
            # Preserve LoRA directory structure
            lora_dir = os.path.dirname(lora.relative_path)
            base_dir = os.path.join(output_dir, lora_dir, test_name)
    else:
        base_dir = output_dir

    # Format filename
    lora_name = "baseline" if is_baseline else lora.name
    filename = filename_template.format(
        lora_name=lora_name,
        image_name=test_name,  # Use test_name
        prompt_name=test_name,
        test_name=test_name,
        seed=seed,
        scale=scale,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_path = os.path.join(base_dir, filename)
    metadata_path = output_path.replace('.mp4', '_metadata.json')

    return TestCase(
        lora=lora,
        test_name=test_name,
        image_path=image_path,
        prompt=prompt,
        seed=seed,
        scale=scale,
        output_path=output_path,
        metadata_path=metadata_path,
        is_baseline=is_baseline,
        is_t2v=is_t2v
    )


def get_test_id(test_case: TestCase) -> str:
    """Generate unique ID for test case."""
    lora_name = "baseline" if test_case.is_baseline else test_case.lora.name
    components = [
        lora_name,
        test_case.test_name,
        str(test_case.seed),
        f"{test_case.scale:.2f}"
    ]
    return hashlib.md5("_".join(components).encode()).hexdigest()


# ============================================================================
# BATCH EXECUTOR
# ============================================================================

class BatchExecutor:
    """Executes batch LoRA testing."""

    def __init__(self, cfg: Dict, state_manager: StateManager):
        self.cfg = cfg
        self.state = state_manager
        self.pipeline = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = cfg['execution']['max_consecutive_errors']

    def initialize_pipeline(self):
        """Initialize the I2V pipeline once."""
        logging.info("Initializing Wan2.2 I2V pipeline...")
        start = time.time()

        self.pipeline = WanI2V(
            config=config,
            checkpoint_dir=self.cfg['paths']['checkpoint_dir'],
            device_id=self.cfg['execution']['device_id'],
            t5_cpu=self.cfg['memory']['t5_cpu'],
            convert_model_dtype=self.cfg['memory']['convert_dtype'],
            lora_verbose=self.cfg['lora']['verbose'],
        )

        elapsed = time.time() - start
        logging.info(f"Pipeline initialized in {elapsed:.1f}s")

    def _calculate_resolution(self, img: Optional[Image.Image]) -> int:
        """Calculate max_area based on resolution strategy."""
        strategy = self.cfg['generation']['resolution_strategy']

        if strategy == "fixed":
            # Use fixed resolution
            width, height = map(int, self.cfg['generation']['fixed_resolution'].split('*'))
            max_area = width * height
            logging.info(f"Using fixed resolution: {width}x{height}")
            return max_area

        elif strategy == "resize":
            # Resize to fit within max dimensions while preserving aspect ratio
            max_width = self.cfg['generation']['max_width']
            max_height = self.cfg['generation']['max_height']

            if img is None:
                # T2V mode: use max dimensions
                max_area = max_width * max_height
                logging.info(f"T2V mode - using max dimensions: {max_width}x{max_height}")
                return max_area

            # Calculate aspect ratio
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height

            # Fit within bounds
            if aspect_ratio > (max_width / max_height):
                # Width is limiting factor
                width = max_width
                height = int(width / aspect_ratio)
            else:
                # Height is limiting factor
                height = max_height
                width = int(height * aspect_ratio)

            # Round to nearest multiple of 8 (required by VAE)
            width = (width // 8) * 8
            height = (height // 8) * 8

            max_area = width * height
            logging.info(f"Resized to fit: {width}x{height} (aspect ratio preserved)")
            return max_area

        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")

    def execute_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create output directory
            os.makedirs(os.path.dirname(test_case.output_path), exist_ok=True)

            # Load image if I2V mode
            img = None
            if not test_case.is_t2v:
                img = Image.open(test_case.image_path).convert('RGB')
                logging.info(f"Loaded image: {test_case.image_path}")

            # Calculate resolution
            max_area = self._calculate_resolution(img)

            # Reinitialize pipeline with/without LoRA
            if test_case.is_baseline:
                logging.info("Loading model (baseline - no LoRA)")
                lora_paths = None
                lora_scales = None
            else:
                logging.info(f"Loading LoRA: {test_case.lora.relative_path}")
                lora_paths = [test_case.lora.path]
                lora_scales = [test_case.scale]

            self.pipeline = WanI2V(
                config=config,
                checkpoint_dir=self.cfg['paths']['checkpoint_dir'],
                device_id=self.cfg['execution']['device_id'],
                t5_cpu=self.cfg['memory']['t5_cpu'],
                convert_model_dtype=self.cfg['memory']['convert_dtype'],
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                lora_verbose=self.cfg['lora']['verbose'],
            )

            # Generate video
            logging.info(f"Generating video (seed={test_case.seed}, {'T2V' if test_case.is_t2v else 'I2V'})...")
            video = self.pipeline.generate(
                input_prompt=test_case.prompt,
                img=img,
                max_area=max_area,
                frame_num=self.cfg['generation']['frames'],
                shift=self.cfg['generation']['shift'],
                sample_solver=self.cfg['generation']['solver'],
                sampling_steps=self.cfg['generation']['steps'],
                guide_scale=self.cfg['generation']['cfg_scale'],
                n_prompt=self.cfg['generation']['negative_prompt'],
                seed=test_case.seed,
                offload_model=self.cfg['memory']['offload_model'],
            )

            # Save video
            save_video(
                tensor=video[None],
                save_file=test_case.output_path,
                fps=24,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )

            # Save metadata
            if self.cfg['output']['save_metadata']:
                self._save_metadata(test_case, time.time() - start_time)

            # Clear cache
            if self.cfg['memory']['clear_cache']:
                torch.cuda.empty_cache()
                gc.collect()

            generation_time = time.time() - start_time
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

            self.consecutive_errors = 0  # Reset on success

            return TestResult(
                test_case=test_case,
                success=True,
                error=None,
                generation_time=generation_time,
                gpu_memory_peak=gpu_memory,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logging.error(f"Test failed: {error_msg}")
            logging.debug(traceback.format_exc())

            self.consecutive_errors += 1

            return TestResult(
                test_case=test_case,
                success=False,
                error=error_msg,
                generation_time=time.time() - start_time,
                gpu_memory_peak=0.0,
                timestamp=datetime.now().isoformat()
            )

    def _save_metadata(self, test_case: TestCase, generation_time: float):
        """Save metadata JSON."""
        metadata = {
            'is_baseline': test_case.is_baseline,
            'is_t2v': test_case.is_t2v,
            'test_name': test_case.test_name,
            'lora_path': test_case.lora.path if test_case.lora else None,
            'lora_relative_path': test_case.lora.relative_path if test_case.lora else None,
            'lora_scale': test_case.scale if not test_case.is_baseline else None,
            'image_path': test_case.image_path,
            'prompt': test_case.prompt,
            'seed': test_case.seed,
            'resolution_strategy': self.cfg['generation']['resolution_strategy'],
            'frames': self.cfg['generation']['frames'],
            'steps': self.cfg['generation']['steps'],
            'cfg_scale': self.cfg['generation']['cfg_scale'],
            'generation_time': generation_time,
            'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'timestamp': datetime.now().isoformat(),
        }

        with open(test_case.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def run(self, test_cases: List[TestCase], dry_run: bool = False):
        """Run all test cases."""
        if dry_run:
            logging.info("DRY RUN MODE - No videos will be generated")
            self._print_plan(test_cases)
            return

        total = len(test_cases)
        completed = 0
        failed = 0
        skipped = 0

        logging.info(f"Starting batch execution: {total} test cases")

        # Initialize pipeline if preload
        if self.cfg['advanced']['preload_model']:
            self.initialize_pipeline()

        # Progress bar
        pbar = tqdm(test_cases, desc="Processing", disable=not self.cfg['logging']['show_progress'])

        for i, test_case in enumerate(pbar):
            test_id = get_test_id(test_case)

            # Check if already completed
            if self.cfg['execution']['skip_existing'] and os.path.exists(test_case.output_path):
                logging.info(f"[{i+1}/{total}] Skipping (exists): {test_case.output_path}")
                skipped += 1
                continue

            if self.cfg['execution']['resume'] and self.state.is_completed(test_id):
                logging.info(f"[{i+1}/{total}] Skipping (completed): {test_id}")
                skipped += 1
                continue

            # Check consecutive errors
            if self.consecutive_errors >= self.max_consecutive_errors:
                logging.error(f"Stopping: {self.consecutive_errors} consecutive errors")
                break

            # Execute test
            logging.info(f"[{i+1}/{total}] Running test: {test_id}")
            result = self.execute_test(test_case)

            if result.success:
                completed += 1
                self.state.mark_completed(test_id)
                logging.info(f"✓ Success ({result.generation_time:.1f}s)")
            else:
                failed += 1
                self.state.mark_failed(test_id, result.error)
                logging.error(f"✗ Failed: {result.error}")

            # Update progress bar
            pbar.set_postfix({
                'ok': completed,
                'fail': failed,
                'skip': skipped
            })

        logging.info("=" * 80)
        logging.info(f"Batch execution complete!")
        logging.info(f"  Completed: {completed}")
        logging.info(f"  Failed: {failed}")
        logging.info(f"  Skipped: {skipped}")
        logging.info("=" * 80)

    def _print_plan(self, test_cases: List[TestCase]):
        """Print execution plan for dry run."""
        print("\n" + "=" * 80)
        print("EXECUTION PLAN")
        print("=" * 80)

        # Count baseline vs LoRA
        baseline_count = sum(1 for tc in test_cases if tc.is_baseline)
        lora_count = len(test_cases) - baseline_count
        t2v_count = sum(1 for tc in test_cases if tc.is_t2v)
        i2v_count = len(test_cases) - t2v_count

        print(f"Total test cases: {len(test_cases)}")
        print(f"  Baseline (no LoRA): {baseline_count}")
        print(f"  With LoRA: {lora_count}")
        print(f"  I2V mode: {i2v_count}")
        print(f"  T2V mode: {t2v_count}")

        print(f"\nFirst 5 test cases:")
        for i, tc in enumerate(test_cases[:5]):
            print(f"\n[{i+1}]")
            if tc.is_baseline:
                print(f"  Mode: BASELINE (no LoRA)")
            else:
                print(f"  LoRA: {tc.lora.relative_path}")
                print(f"  Scale: {tc.scale}")
            print(f"  Test: {tc.test_name}")
            print(f"  Type: {'T2V' if tc.is_t2v else 'I2V'}")
            if tc.image_path:
                print(f"  Image: {tc.image_path}")
            print(f"  Prompt: {tc.prompt[:60]}...")
            print(f"  Seed: {tc.seed}")
            print(f"  Output: {tc.output_path}")
        if len(test_cases) > 5:
            print(f"\n... and {len(test_cases) - 5} more")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def setup_logging(cfg: Dict):
    """Setup logging configuration."""
    log_file = cfg['paths']['log_file']
    level = getattr(logging, cfg['logging']['level'])

    # File handler
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(cfg['logging']['format']))

    # Console handler
    handlers = [file_handler]
    if cfg['logging']['console']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(cfg['logging']['format']))
        handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(description="Batch LoRA Testing for Wan2.2")
    parser.add_argument(
        "--config",
        type=str,
        default="batch_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override with CLI args
    if args.dry_run:
        cfg['execution']['dry_run'] = True
    if args.resume:
        cfg['execution']['resume'] = True

    # Setup logging
    setup_logging(cfg)

    logging.info("=" * 80)
    logging.info("Wan2.2 Batch LoRA Testing Framework")
    logging.info("=" * 80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Output: {cfg['paths']['output_dir']}")

    # Initialize state manager
    state = StateManager(cfg['paths']['state_file'])

    # Discover LoRAs
    loras = discover_loras(
        lora_root=cfg['paths']['lora_root'],
        lora_filename=cfg['paths']['lora_filename'],
        max_depth=cfg['lora_discovery']['max_depth'],
        exclude_patterns=cfg['lora_discovery']['exclude_patterns'],
        filter_subdirs=cfg['lora_discovery'].get('filter_subdirs')
    )

    if not loras:
        logging.error("No LoRAs discovered! Check your configuration.")
        return 1

    # Generate test cases
    test_cases = generate_test_cases(
        loras,
        cfg,
        cfg['paths']['output_dir']
    )

    if not test_cases:
        logging.error("No test cases generated!")
        return 1

    # Execute
    executor = BatchExecutor(cfg, state)
    executor.run(test_cases, dry_run=cfg['execution']['dry_run'])

    logging.info("All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

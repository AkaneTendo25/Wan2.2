# Batch LoRA Testing Framework

Automated framework for testing hundreds of LoRAs with multiple configurations. Designed for multi-day runs with resume capability.

## Table of Contents

- [Quick Start](#quick-start)
- [What It Does](#what-it-does)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Common Workflows](#common-workflows)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

### Required Files

- `batch_inference.py` - Main script
- `batch_config.yaml` - Configuration file
- Python packages: `pyyaml`, `torch`, `PIL`, `tqdm`, `safetensors`

### Installation

```bash
pip install pyyaml
# Other dependencies should already be installed with Wan2.2
```

### Basic Usage

```bash
# 1. Edit configuration
nano batch_config.yaml

# 2. Test run (dry run - no videos generated)
python batch_inference.py --config batch_config.yaml --dry-run

# 3. Start batch processing
python batch_inference.py --config batch_config.yaml --resume

# 4. Monitor progress
tail -f batch_inference.log
```

### Resume After Interruption

```bash
python batch_inference.py --config batch_config.yaml --resume
```

The framework automatically tracks progress and skips completed tests.

---

## What It Does

### Workflow

1. **Discovers LoRAs**: Recursively finds all `adapter_model.safetensors` in your training output directory
2. **Generates Baseline** (optional): Creates no-LoRA outputs for comparison
3. **Tests All Combinations**: LoRAs × Tests × Seeds × Scales
4. **Saves Results**: Videos + JSON metadata with all parameters
5. **Tracks Progress**: Automatic resume on crash/interruption
6. **Organizes Output**: Flexible directory structures

### Example Scenario

You have:
- 50 LoRAs from training experiments
- 3 test configurations (2 I2V + 1 T2V)
- 3 random seeds for consistency checking
- Want baseline + LoRA at scales [0.8, 1.0, 1.2]

**Total videos:**
- Baseline: 3 tests × 3 seeds = **9 videos**
- LoRA: 50 LoRAs × 3 tests × 3 seeds × 3 scales = **1,350 videos**
- **Grand total: 1,359 videos**

At ~5 min/video = **~113 hours (~4.7 days)**

The framework handles this automatically with progress tracking, auto-resume, and metadata for each video.

---

## Configuration

### Minimal Example

```yaml
paths:
  checkpoint_dir: "./Wan2.2-I2V-A14B"
  lora_root: "./.output"
  output_dir: "./batch_results"

test_inputs:
  tests:
    - name: "portrait"
      image: "./portrait.jpg"
      prompt: "Cinematic portrait with dramatic lighting"

    - name: "abstract_t2v"
      image: null  # T2V mode
      prompt: "Abstract flowing patterns"

generation:
  resolution_strategy: "resize"
  max_width: 1280
  max_height: 720
  frames: 81
  steps: 40
  seeds: [42, 123, 456]

advanced:
  include_baseline: true
```

### Key Configuration Sections

#### 1. Paths

```yaml
paths:
  checkpoint_dir: "./Wan2.2-I2V-A14B"
  lora_root: "./.output"              # Your training output directory
  lora_filename: "adapter_model.safetensors"
  output_dir: "./batch_results"
  state_file: "./batch_state.json"    # Progress tracking
  log_file: "./batch_inference.log"
```

#### 2. Test Inputs

Each test combines image (optional) + prompt:

```yaml
test_inputs:
  tests:
    # I2V example
    - name: "portrait_cinematic"
      image: "./images/portrait.jpg"
      prompt: "Cinematic portrait with dramatic lighting"

    # T2V example (no image)
    - name: "abstract_motion"
      image: null
      prompt: "Abstract flowing patterns with vibrant colors"
```

#### 3. Resolution Strategies

**Fixed Resolution** (ignores aspect ratio):
```yaml
generation:
  resolution_strategy: "fixed"
  fixed_resolution: "1280*720"
```

**Adaptive Resize** (preserves aspect ratio):
```yaml
generation:
  resolution_strategy: "resize"
  max_width: 1280
  max_height: 720
```

Automatically fits images within bounds while preserving aspect ratio.

#### 4. LoRA Discovery

```yaml
lora_discovery:
  max_depth: 5
  exclude_patterns:
    - "*/checkpoint-*"   # Skip training checkpoints
    - "*/.git/*"
  filter_subdirs: null   # null = all, or ["exp1", "exp2"]
```

#### 5. Output Organization

```yaml
output:
  structure: "preserve"  # Options: preserve, hierarchical, by_lora, by_test, flat
  filename_template: "video_s{seed}_scale{scale}.mp4"
  save_metadata: true
```

**Structure Options:**

- **`preserve`** (recommended): Mirrors LoRA directory structure
  ```
  .output/exp1/run1/adapter_model.safetensors
  → batch_results/exp1/run1/test_name/video.mp4
  ```

- **`hierarchical`**: Group by LoRA then test
  ```
  batch_results/lora_name/test_name/video.mp4
  ```

- **`by_test`**: Group by test name
  ```
  batch_results/test_name/lora_name_video.mp4
  ```

- **`by_lora`**: Group by LoRA
  ```
  batch_results/lora_name/video.mp4
  ```

- **`flat`**: All in one directory

#### 6. Baseline Generation

```yaml
advanced:
  include_baseline: true
  baseline_output_dir: "baseline"
```

Generates no-LoRA outputs first for A/B comparison:
```
batch_results/
├── baseline/          # No LoRA
│   ├── test1/
│   └── test2/
└── exp1/              # With LoRA
    └── run1/
```

#### 7. Memory Optimization

```yaml
memory:
  offload_model: true      # Offload to CPU between generations
  convert_dtype: true      # Use FP16/BF16
  t5_cpu: false           # Keep T5 on GPU (faster)
  clear_cache: true       # Clear CUDA cache
```

For limited VRAM, enable all optimizations + reduce resolution.

#### 8. Execution Control

```yaml
execution:
  skip_existing: true              # Don't regenerate existing videos
  resume: true                     # Resume from checkpoint
  max_consecutive_errors: 5        # Stop after N errors
  retry_on_error: true
  max_retries: 2
```

---

## Output Structure

### Example Output (preserve mode)

```
batch_results/
├── baseline/                         # Baseline (no LoRA)
│   ├── portrait/
│   │   ├── video_s42_scale1.0.mp4
│   │   ├── video_s42_scale1.0_metadata.json
│   │   ├── video_s123_scale1.0.mp4
│   │   └── video_s123_scale1.0_metadata.json
│   └── abstract_t2v/
├── experiment1/                      # Mirrors .output structure
│   ├── run_lr1e-4/
│   │   ├── portrait/
│   │   │   ├── video_s42_scale0.8.mp4
│   │   │   ├── video_s42_scale0.8_metadata.json
│   │   │   ├── video_s42_scale1.0.mp4
│   │   │   ├── video_s42_scale1.2.mp4
│   │   │   └── ...
│   │   └── abstract_t2v/
│   └── run_lr5e-5/
│       └── ...
└── experiment2/
    └── ...
```

### Metadata JSON

Each video has accompanying metadata:

```json
{
  "is_baseline": false,
  "is_t2v": false,
  "test_name": "portrait_cinematic",
  "lora_path": "./.output/exp1/run1/adapter_model.safetensors",
  "lora_relative_path": "exp1/run1/adapter_model.safetensors",
  "lora_scale": 1.0,
  "image_path": "./images/portrait.jpg",
  "prompt": "Cinematic portrait with dramatic lighting",
  "seed": 42,
  "resolution_strategy": "resize",
  "frames": 81,
  "steps": 40,
  "cfg_scale": 5.0,
  "generation_time": 324.5,
  "gpu_memory_gb": 22.3,
  "timestamp": "2026-01-20T15:30:45.123456"
}
```

Use metadata for:
- Filtering results by parameters
- Tracking generation performance
- Reproducing specific results

### Progress Tracking

**State file (`batch_state.json`):**
```json
{
  "completed": ["hash1", "hash2", ...],
  "failed": [
    {"test_id": "hash3", "error": "CUDA OOM", "timestamp": "..."}
  ],
  "start_time": "2026-01-20T10:00:00",
  "last_update": "2026-01-20T15:30:45"
}
```

**Log file (`batch_inference.log`):**
Contains detailed execution log with timestamps.

---

## Common Workflows

### 1. A/B Testing Multiple LoRAs

Compare all LoRAs against baseline:

```yaml
advanced:
  include_baseline: true

generation:
  seeds: [42]  # Fixed seed for fair comparison

output:
  structure: "by_test"  # Easy side-by-side comparison
```

**Result**: All LoRAs + baseline for each test in same directory.

### 2. Finding Optimal LoRA Scale

Test different strengths:

```yaml
lora:
  test_scales: [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

generation:
  seeds: [42]  # Single seed

output:
  structure: "hierarchical"  # Group by LoRA
```

**Result**: All scales for visual comparison.

### 3. Consistency Testing

Check if LoRA produces consistent results:

```yaml
generation:
  seeds: [42, 123, 456, 789, 1000]  # Multiple seeds

output:
  structure: "by_lora"
```

**Result**: Multiple videos per LoRA to check variance.

### 4. Multi-Prompt Testing

Test LoRA behavior with different prompts:

```yaml
test_inputs:
  tests:
    - name: "portrait_cinematic"
      image: "./portrait.jpg"
      prompt: "Cinematic portrait with dramatic lighting"

    - name: "portrait_natural"
      image: "./portrait.jpg"
      prompt: "Natural lighting, soft focus"

    - name: "portrait_artistic"
      image: "./portrait.jpg"
      prompt: "Artistic style, vibrant colors"
```

**Result**: See how each LoRA handles different styles.

### 5. I2V + T2V Mixed Testing

```yaml
test_inputs:
  tests:
    - name: "i2v_test"
      image: "./image.jpg"
      prompt: "..."

    - name: "t2v_test"
      image: null  # T2V mode
      prompt: "..."
```

**Result**: Test LoRA on both modes.

---

## Monitoring & Troubleshooting

### Monitoring Progress

```bash
# Real-time log
tail -f batch_inference.log

# Count completed
grep "✓ Success" batch_inference.log | wc -l

# Check state
cat batch_state.json | jq '.completed | length'

# View errors
grep "✗ Failed" batch_inference.log

# Estimate remaining
# (Total - Completed) × Average time per video
```

### Common Issues

#### Out of Memory (OOM)

**Solution:**
```yaml
memory:
  offload_model: true
  convert_dtype: true
  t5_cpu: true  # Move T5 to CPU

generation:
  resolution_strategy: "resize"
  max_width: 1024      # Reduce from 1280
  max_height: 576      # Reduce from 720
  frames: 81           # Don't use 161+
  steps: 30            # Reduce from 40
```

#### Slow Generation

**Optimize:**
- Reduce `steps` from 40 to 20-30
- Use `offload_model: false` if you have enough VRAM
- Use lower resolution
- Check disk speed (use SSD for output)

**Typical Performance:**
- RTX 4090: ~5-8 min per 720p/81 frame video
- A100 40GB: ~3-5 min
- A100 80GB: ~2-3 min

#### Script Crashes

**Auto-restart (Linux/Mac):**
```bash
#!/bin/bash
# save as run.sh
while true; do
    python batch_inference.py --config batch_config.yaml --resume
    if [ $? -eq 0 ]; then break; fi
    echo "Restarting in 10 seconds..."
    sleep 10
done
```

**Manual resume:**
```bash
python batch_inference.py --config batch_config.yaml --resume
```

#### Wrong LoRAs Discovered

**Filter discovery:**
```yaml
lora_discovery:
  exclude_patterns:
    - "*/checkpoint-*"      # Exclude training checkpoints
    - "*/temp/*"
    - "*/_old/*"

  filter_subdirs: ["experiment1", "experiment2"]  # Only these
```

#### Videos Already Exist

```yaml
execution:
  skip_existing: true   # Skip if video exists
  # or
  skip_existing: false  # Regenerate all
```

---

## Advanced Topics

### Multiple LoRA Scales per LoRA

Test each LoRA at different strengths:

```yaml
lora:
  test_scales: [0.5, 0.8, 1.0, 1.2]
```

Generates 4 videos per LoRA/test/seed combination.

### Parallel Processing (Experimental)

```yaml
advanced:
  parallel: true
  num_workers: 2
```

**Warning**: May cause OOM. Test carefully.

### Checkpointing

```yaml
advanced:
  checkpoint_interval: 10  # Save state every 10 generations
```

State saved in `batch_state.json` for resume.

### Custom Filename Templates

```yaml
output:
  filename_template: "{test_name}_{lora_name}_s{seed}_scale{scale}.mp4"
```

Available variables:
- `{lora_name}`
- `{test_name}`
- `{seed}`
- `{scale}`
- `{timestamp}`

### Timeout Control

```yaml
execution:
  generation_timeout: 3600  # 1 hour per video, null = no timeout
```

Prevents hanging on problematic generations.

---

## Performance Estimates

### Time Calculation

**Formula**: Videos × Time per video = Total time

**Example:**
- 50 LoRAs × 3 tests × 3 seeds × 3 scales = 1,350 videos
- At 5 min/video = 6,750 minutes = **112.5 hours (~4.7 days)**

### Optimization Tips

1. **Start small**: Test with 2-3 LoRAs first
2. **Use single seed**: For initial screening
3. **Lower steps**: 20-30 instead of 40 for drafts
4. **Enable baseline**: Essential for comparison
5. **Fixed seeds**: Use same seed across LoRAs for fair A/B
6. **Organize output**: Use `preserve` structure to maintain hierarchy
7. **Monitor logs**: Check periodically for errors
8. **Resume enabled**: Always use `--resume` for long runs

---

## Best Practices

### Before Starting

1. **Dry run first**: Always check the plan
   ```bash
   python batch_inference.py --dry-run
   ```

2. **Test small batch**: Verify setup with 2-3 LoRAs

3. **Check disk space**: ~500MB per video (depends on resolution/frames)

4. **Verify LoRA discovery**: Check dry run output for correct LoRAs

### During Execution

1. **Monitor logs**: `tail -f batch_inference.log`

2. **Check progress**: Periodically review `batch_state.json`

3. **Backup state file**: Copy `batch_state.json` occasionally

4. **Watch for errors**: Check for repeated failures

### After Completion

1. **Review metadata**: Filter results by parameters

2. **Compare baseline**: Essential reference point

3. **Identify patterns**: Which LoRAs/scales work best

4. **Archive results**: Keep metadata for reproducibility

---

## Requirements

- **GPU**: 24GB+ VRAM recommended (RTX 4090 or better)
- **RAM**: 32GB+ (64GB for large batches)
- **Disk**: ~500MB per video × number of videos
- **Python**: 3.8+
- **PyTorch**: 2.4+
- **Dependencies**: `pyyaml`, `torch`, `PIL`, `tqdm`, `safetensors`

---

## Command Reference

```bash
# Dry run (plan without execution)
python batch_inference.py --config CONFIG.yaml --dry-run

# Normal run
python batch_inference.py --config CONFIG.yaml

# Resume from checkpoint
python batch_inference.py --config CONFIG.yaml --resume

# Custom config file
python batch_inference.py --config my_config.yaml --resume
```

---

## Support

- Configuration: See extensive comments in `batch_config.yaml`
- Dry run: `python batch_inference.py --dry-run`
- GitHub: https://github.com/Wan-Video/Wan2.2
- Discord: https://discord.gg/AKNgpMK4Yj

---

**Ready to start?**

```bash
# 1. Edit configuration
nano batch_config.yaml

# 2. Verify plan
python batch_inference.py --dry-run

# 3. Start batch processing
python batch_inference.py --resume
```

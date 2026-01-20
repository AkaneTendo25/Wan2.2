# Quick Start: I2V Inference with LoRA

Three ways to run I2V inference with LoRA support.

## Method 1: Using generate.py (Main Script)

Use the main script with full control over all parameters:

```bash
# Basic I2V without LoRA
python generate.py \
  --task i2v-A14B \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --offload_model True

# With LoRA
python generate.py \
  --task i2v-A14B \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora_path ./loras/style.safetensors \
  --lora_scale 1.0 \
  --lora_verbose \
  --offload_model True

# With expert-specific LoRAs
python generate.py \
  --task i2v-A14B \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --high_noise_lora_path ./loras/composition.safetensors \
  --low_noise_lora_path ./loras/details.safetensors \
  --lora_verbose \
  --offload_model True
```

## Method 2: Using test_i2v_lora.py (Standalone Test Script)

Dedicated script for testing and tuning:

```bash
# Basic usage
python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora ./loras/style.safetensors \
  --output ./output.mp4

# Full options
python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora ./loras/style.safetensors --lora_scale 0.8 \
  --lora ./loras/motion.safetensors --lora_scale 1.0 \
  --prompt "Cinematic video with smooth motion" \
  --output ./output.mp4 \
  --resolution 1280*720 \
  --frames 81 \
  --steps 40 \
  --cfg_scale 5.0 \
  --seed 42 \
  --verbose

# With expert-specific LoRAs
python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --high_noise_lora ./loras/composition.safetensors \
  --low_noise_lora ./loras/details.safetensors \
  --output ./output.mp4 \
  --verbose
```

**Help:**
```bash
python test_i2v_lora.py --help
```

## Method 3: Using simple_test.py (Minimal Script)

Easiest for quick tests - just edit the variables in the file:

1. Open `simple_test.py` in a text editor
2. Edit these variables:
   ```python
   CHECKPOINT_DIR = "./Wan2.2-I2V-A14B"
   INPUT_IMAGE = "./input.jpg"
   OUTPUT_VIDEO = "./output.mp4"

   LORA_PATHS = ["./loras/style.safetensors"]
   LORA_SCALES = [1.0]

   PROMPT = "Your prompt here"
   FRAMES = 81
   STEPS = 40
   ```
3. Run:
   ```bash
   python simple_test.py
   ```

This is the fastest way to iterate - no command line arguments needed!

## Comparison

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `generate.py` | Production, full control | All features, same as official | Long command lines |
| `test_i2v_lora.py` | Testing, parameter tuning | Clean CLI, LoRA-focused | Standalone script |
| `simple_test.py` | Quick iterations | No CLI args, fastest to edit | Need to edit file |

## Common Parameters

### LoRA Scaling Recommendations

| Scale | Effect | Use Case |
|-------|--------|----------|
| 0.3-0.5 | Subtle | Minor style adjustments |
| 0.7-0.8 | Moderate | Balanced effect |
| 1.0 | Full | Standard recommended |
| 1.2-1.5 | Strong | Emphasize LoRA (may reduce quality) |

### Resolution Options

- `480*832` - Low resolution, faster (use `--sample_shift 3.0`)
- `720*1280` - Portrait mode
- `1280*720` - High resolution (default, use `--sample_shift 5.0`)

### Frame Numbers

Must be `4n+1`:
- `81` - ~3.4 seconds @ 24fps (default)
- `161` - ~6.7 seconds @ 24fps
- `241` - ~10 seconds @ 24fps

### Memory Optimization

If you get Out of Memory errors:

```bash
# Enable all optimizations
python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --offload \
  --fp16 \
  --t5_cpu \
  --resolution 480*832 \
  --frames 81
```

## Getting LoRA Weights

### Pre-trained LoRAs:

```bash
# Download LightX2V distilled LoRAs (4-step fast inference)
huggingface-cli download lightx2v/Wan2.2-Distill-Loras --local-dir ./loras/
```

### Training Your Own:

See [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training) for training scripts.

## Troubleshooting

### LoRA not loading

```bash
# Enable verbose mode to see what's happening
python test_i2v_lora.py ... --verbose

# Check logs for:
# "[LoRA] Applied to model: matched=XXX"
# If matched=0, LoRA is incompatible
```

### Out of Memory

```bash
# 1. Enable all memory optimizations
--offload --fp16 --t5_cpu

# 2. Reduce resolution
--resolution 480*832

# 3. Reduce frames
--frames 81

# 4. Use fewer LoRAs
# Use 1-2 LoRAs instead of many
```

### Slow generation

```bash
# 1. Reduce steps
--steps 20  # Instead of 40

# 2. Use distilled LoRAs for 4-step inference
--lora ./loras/wan2.2-lightning-lora.safetensors --steps 4

# 3. Lower resolution
--resolution 480*832
```

## Example Workflow

Here's a typical workflow for testing LoRA:

```bash
# 1. Quick test with simple_test.py
# Edit simple_test.py, set LORA_PATHS
python simple_test.py

# 2. Fine-tune parameters with test_i2v_lora.py
python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora ./loras/style.safetensors --lora_scale 0.5 \
  --output ./test_05.mp4

python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora ./loras/style.safetensors --lora_scale 0.8 \
  --output ./test_08.mp4

python test_i2v_lora.py \
  --checkpoint ./Wan2.2-I2V-A14B \
  --image ./input.jpg \
  --lora ./loras/style.safetensors --lora_scale 1.2 \
  --output ./test_12.mp4

# 3. Compare results and choose best scale
# 4. Use generate.py for final production run
```

## Next Steps

- See [README.md](README.md) for full documentation
- Training LoRAs: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- Pre-trained LoRAs: [HuggingFace](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- Join [Discord](https://discord.gg/AKNgpMK4Yj) for community support

#!/bin/bash
# Launcher script for batch LoRA testing
# Usage: ./run_batch.sh [options]

set -e

# Default config
CONFIG="batch_config.yaml"
DRY_RUN=false
RESUME=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --config FILE    Configuration file (default: batch_config.yaml)"
      echo "  --dry-run        Print execution plan without running"
      echo "  --resume         Resume from last checkpoint"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Wan2.2 Batch LoRA Testing"
echo "========================================"
echo "Config: $CONFIG"
echo "Dry run: $DRY_RUN"
echo "Resume: $RESUME"
echo "========================================"
echo ""

# Build command
CMD="python batch_inference.py --config $CONFIG"

if [ "$DRY_RUN" = true ]; then
  CMD="$CMD --dry-run"
fi

if [ "$RESUME" = true ]; then
  CMD="$CMD --resume"
fi

# Run with auto-restart on crash (useful for multi-day runs)
MAX_RESTARTS=5
restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
  echo "Starting batch inference (attempt $((restart_count + 1))/$MAX_RESTARTS)..."
  echo "Command: $CMD"
  echo ""

  # Run and capture exit code
  set +e
  $CMD
  exit_code=$?
  set -e

  if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Batch inference completed successfully!"
    echo "========================================"
    exit 0
  else
    echo ""
    echo "========================================"
    echo "✗ Batch inference failed with exit code: $exit_code"
    echo "========================================"

    if [ "$RESUME" = true ]; then
      restart_count=$((restart_count + 1))
      if [ $restart_count -lt $MAX_RESTARTS ]; then
        echo "Auto-restarting with resume in 10 seconds..."
        sleep 10
      else
        echo "Max restarts reached. Exiting."
        exit $exit_code
      fi
    else
      echo "Resume not enabled. Exiting."
      exit $exit_code
    fi
  fi
done

echo "Max restart attempts reached. Exiting."
exit 1

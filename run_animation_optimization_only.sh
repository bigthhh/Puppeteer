#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/animation:${PYTHONPATH:-}"

SEQ_NAME="${1:-}"
SAVE_NAME="${2:-${SEQ_NAME:+${SEQ_NAME}_demo}}"
ITER="${ITER:-200}"
IMG_SIZE="${IMG_SIZE:-960}"
INPUT_PATH="${INPUT_PATH:-$SCRIPT_DIR/examples}"
SAVE_PATH="${SAVE_PATH:-$SCRIPT_DIR/results/animation}"

check_flow_ready() {
  local seq_name="$1"
  local flow_dir="$INPUT_PATH/$seq_name/flow"
  if [ ! -d "$flow_dir" ]; then
    echo "Missing flow directory: $flow_dir"
    echo "Please generate optical flow first (save_flow.py)."
    exit 1
  fi
  if ! ls "$flow_dir"/flow_*.flo >/dev/null 2>&1; then
    echo "No .flo files found in: $flow_dir"
    echo "Please generate optical flow first (save_flow.py)."
    exit 1
  fi
}

run_optimization() {
  local seq_name="$1"
  local save_name="$2"
  shift 2
  python optimization.py \
    --save_path "$SAVE_PATH" \
    --iter "$ITER" \
    --input_path "$INPUT_PATH" \
    --img_size "$IMG_SIZE" \
    --seq_name "$seq_name" \
    --save_name "$save_name" \
    "$@"
}

mkdir -p "$SAVE_PATH"

cd "$SCRIPT_DIR/animation"

if [ -n "$SEQ_NAME" ]; then
  check_flow_ready "$SEQ_NAME"
  echo "Running optimization for: $SEQ_NAME (save_name: $SAVE_NAME)..."
  shift 2 2>/dev/null || true
  run_optimization "$SEQ_NAME" "$SAVE_NAME" "$@"
else
  check_flow_ready "spiderman"
  check_flow_ready "deer"
  echo "Running optimization only (skip frame extraction / flow generation)..."
  run_optimization "spiderman" "spiderman_demo"
  run_optimization "deer" "deer_demo" \
    --smooth_weight 1 \
    --main_renderer front_left \
    --additional_renderers "right,front_right,back_right"
fi

echo "Optimization finished. Results saved to: $SAVE_PATH"

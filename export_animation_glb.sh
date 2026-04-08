#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$SCRIPT_DIR/.env"
  set +a
fi

SEQ_NAME="${1:-}"
SAVE_NAME="${2:-}"
OUT_GLB="${3:-}"
MOTION_KIND="${MOTION_KIND:-raw}" # raw | smoothed
APPLY_ROOT_MOTION="${APPLY_ROOT_MOTION:-1}"
ROOT_CORRECTION_DEG="${ROOT_CORRECTION_DEG:-90,0,0}" # e.g. "90,0,0" for +90deg around X
ROOT_BONE_CORRECTION_DEG="${ROOT_BONE_CORRECTION_DEG:-0,0,0}" # relative correction between mesh and skeleton

if [ -z "$SEQ_NAME" ] || [ -z "$SAVE_NAME" ] || [ -z "$OUT_GLB" ]; then
  echo "Usage: bash export_animation_glb.sh <seq_name> <save_name> <out_glb_path>"
  echo "Example: bash export_animation_glb.sh deer deer_demo results/animation/deer/deer_demo/deer.glb"
  exit 1
fi

BLENDER_EXEC="${BLENDER_BIN:-${BLENDER_PATH:-}}"
if [ -z "$BLENDER_EXEC" ]; then
  BLENDER_EXEC="$(command -v blender || true)"
fi
if [ -z "$BLENDER_EXEC" ] || [ ! -x "$BLENDER_EXEC" ]; then
  echo "Blender not found. Set BLENDER_BIN or BLENDER_PATH in .env, or ensure 'blender' is in PATH."
  exit 1
fi

MESH_OBJ="$SCRIPT_DIR/examples/$SEQ_NAME/objs/mesh.obj"
RIG_TXT="$SCRIPT_DIR/examples/$SEQ_NAME/objs/rig.txt"
MOTION_DIR="$SCRIPT_DIR/results/animation/$SEQ_NAME/$SAVE_NAME/$MOTION_KIND"

for p in "$MESH_OBJ" "$RIG_TXT" "$MOTION_DIR/local_quats.pt" "$MOTION_DIR/root_quats.pt" "$MOTION_DIR/root_pos.pt"; do
  if [ ! -f "$p" ]; then
    echo "Missing required file: $p"
    exit 1
  fi
done

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

python - "$MOTION_DIR" "$TMP_DIR" <<'PY'
import os
import sys
import numpy as np
import torch

motion_dir = sys.argv[1]
tmp_dir = sys.argv[2]

for name in ("local_quats", "root_quats", "root_pos"):
    pt_path = os.path.join(motion_dir, f"{name}.pt")
    arr = torch.load(pt_path, map_location="cpu")
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    np.save(os.path.join(tmp_dir, f"{name}.npy"), arr)
PY

mkdir -p "$(dirname "$OUT_GLB")"

"$BLENDER_EXEC" --background --python-exit-code 1 --python "$SCRIPT_DIR/animation/utils/blender_export_glb.py" -- \
  --mesh_obj "$MESH_OBJ" \
  --rig_txt "$RIG_TXT" \
  --local_quats_npy "$TMP_DIR/local_quats.npy" \
  --root_quats_npy "$TMP_DIR/root_quats.npy" \
  --root_pos_npy "$TMP_DIR/root_pos.npy" \
  --out_glb "$OUT_GLB" \
  --apply_root_motion "$APPLY_ROOT_MOTION" \
  --root_correction_deg="$ROOT_CORRECTION_DEG" \
  --root_bone_correction_deg="$ROOT_BONE_CORRECTION_DEG"

if [ ! -f "$OUT_GLB" ]; then
  echo "GLB export failed: output file not found: $OUT_GLB"
  exit 1
fi

echo "Exported GLB: $OUT_GLB"

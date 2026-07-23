#!/bin/bash
# Stage 2b: GT-tuned oracle force cache, one array task per scene.
# For each scene, computes FTTC+L2 and FISTA+L1 GT-tuned oracle traction maps for
# every cached displacement field. Submit from $CODE after `source $STAGE/env.sh`.
# Runs on CPU (partition=common). The L1 FISTA solve CAN use a CuPy GPU backend, but
# at 512² it is launch-bound and only ~3× over CPU — not worth holding a scarce GPU
# node for (see _resolve_backend / docs/specs/forward-solver-pcg.md). FWD_DEVICE=cpu
# below keeps it on numpy/scipy; the FTTC part is CPU-only numba regardless. For the
# GPU variant, use partition=gpu + --gres=gpu:1 and FWD_DEVICE=cuda (fails loud if the
# venv lacks a matching cupy-cuda12x wheel).
#SBATCH --job-name=tfm_force
#SBATCH --partition=common
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/force_%A_%a.out
#SBATCH --error=../logs/force_%A_%a.err
set -euo pipefail
export TMPDIR="${TMPDIR:-/local/scratch/tmp}"
export NUMBA_CACHE_DIR="$TMPDIR"
export FWD_DEVICE=cpu           # CPU partition: numpy/scipy backend (no GPU here)
: "${STAGE:?source env.sh first}" "${CODE:?}"
cd "$CODE"
# Flat array over every condition x scene: ITEMS[i] = "<condition>/<scene_id>".
mapfile -t ITEMS < <(cd "$STAGE/scenes" && ls -d ${SCENE_GLOB:-*/*} 2>/dev/null)
ITEM="${ITEMS[$SLURM_ARRAY_TASK_ID]}"
COND="${ITEM%%/*}"; SCENE="${ITEM##*/}"
echo "task $SLURM_ARRAY_TASK_ID -> $COND / $SCENE  (oracle FTTC+L2 & FISTA+L1)"
python build_force_cache.py --stage "$STAGE" --condition "$COND" --scene "$SCENE" ${OVERWRITE:+--overwrite}

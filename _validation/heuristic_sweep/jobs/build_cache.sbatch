#!/bin/bash
# Stage 1: displacement cache, one array task per scene.
# Submit from $CODE after `source $STAGE/env.sh` (submit.sh does the array math).
# Runs on CPU (common): PIV + force use CPU paths, device="auto" -> cpu, no torch.
# For a GPU run instead: partition=gpu, qos=gpu, add --gres=gpu:1, install cuda torch.
#SBATCH --job-name=tfm_cache
#SBATCH --partition=common
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/cache_%A_%a.out
#SBATCH --error=../logs/cache_%A_%a.err
set -euo pipefail
export TMPDIR="${TMPDIR:-/local/scratch/tmp}"   # node /tmp is capped at 500 MB
export NUMBA_CACHE_DIR="$TMPDIR"
: "${STAGE:?source env.sh first}" "${CODE:?}" "${CONDITION:?}" "${METHOD:?}"
cd "$CODE"
mapfile -t SCENES < <(ls -1 "$STAGE/scenes/$CONDITION")
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> scene $SCENE"
python build_cache.py --stage "$STAGE" --condition "$CONDITION" --method "$METHOD" --scene "$SCENE"

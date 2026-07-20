#!/bin/bash
# Stage 1: displacement cache, one array task per scene.
# Submit from $CODE after `source $STAGE/env.sh` (submit.sh does the array math).
# All methods (PIV/iLK/FFD) have GPU paths; device="auto" uses the allocated GPU.
#SBATCH --job-name=tfm_cache
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/cache_%A_%a.out
#SBATCH --error=../logs/cache_%A_%a.err
set -euo pipefail
: "${STAGE:?source env.sh first}" "${CODE:?}" "${CONDITION:?}" "${METHOD:?}"
cd "$CODE"
mapfile -t SCENES < <(ls -1 "$STAGE/scenes/$CONDITION")
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> scene $SCENE"
python build_cache.py --stage "$STAGE" --condition "$CONDITION" --method "$METHOD" --scene "$SCENE"

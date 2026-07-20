#!/bin/bash
# Stage 2: elastic-net force sweep over the cache. One array task per scene;
# each writes its own CSV shard (results/sweep_<condition>_<scene>.csv), so tasks
# never contend on one file. Concatenate shards afterwards.
#SBATCH --job-name=tfm_sweep
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/sweep_%A_%a.out
#SBATCH --error=../logs/sweep_%A_%a.err
set -euo pipefail
export TMPDIR="${TMPDIR:-/local/scratch/tmp}"   # node /tmp is capped at 500 MB
export NUMBA_CACHE_DIR="$TMPDIR"
: "${STAGE:?source env.sh first}" "${CODE:?}"
cd "$CODE"
# Flat array over every condition x scene: ITEMS[i] = "<condition>/<scene_id>".
mapfile -t ITEMS < <(cd "$STAGE/scenes" && ls -d */* 2>/dev/null)
ITEM="${ITEMS[$SLURM_ARRAY_TASK_ID]}"
COND="${ITEM%%/*}"; SCENE="${ITEM##*/}"
echo "task $SLURM_ARRAY_TASK_ID -> $COND / $SCENE"
python sweep_forces.py --stage "$STAGE" --condition "$COND" --scene "$SCENE" \
    --out "$STAGE/results/sweep_${COND}_${SCENE}.csv"

#!/bin/bash
# Stage 2 (CPU): elastic-net force sweep over the displacement cache, on the
# `common` partition. The inversion is CPU-bound (TotalCPU~=Elapsed, ~90 MB RSS,
# a small FFTC solve at the GT grid), so it needs no GPU -- running it on common
# frees the scarce gpu partition and lets us fan out wide (hundreds of idle cores).
# One array task per (condition x scene); each writes its own CSV shard, so tasks
# never contend. Submit with --dependency=aftercorr:<cache_jobid> (submit.sh does
# this) so each scene's sweep starts the instant its cache lands on the gpu side.
#SBATCH --job-name=tfm_sweep
#SBATCH --partition=common
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/sweep_%A_%a.out
#SBATCH --error=../logs/sweep_%A_%a.err
set -euo pipefail
export TMPDIR="${TMPDIR:-/local/scratch/tmp}"   # node /tmp is capped at 500 MB
export NUMBA_CACHE_DIR="$TMPDIR"
# Pin math-library threads to the allocation so N concurrent tasks don't oversubscribe.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""                  # force fwd_device="auto" -> cpu, deterministically
: "${STAGE:?source env.sh first}" "${CODE:?}"
cd "$CODE"
# Flat array over every condition x scene: ITEMS[i] = "<condition>/<scene_id>".
# MUST match build_cache.sbatch's ordering exactly for aftercorr to pair correctly.
mapfile -t ITEMS < <(cd "$STAGE/scenes" && ls -d */* 2>/dev/null)
ITEM="${ITEMS[$SLURM_ARRAY_TASK_ID]}"
COND="${ITEM%%/*}"; SCENE="${ITEM##*/}"
echo "task $SLURM_ARRAY_TASK_ID -> $COND / $SCENE"
python sweep_forces.py --stage "$STAGE" --condition "$COND" --scene "$SCENE" \
    --out "$STAGE/results/sweep_${COND}_${SCENE}.csv"

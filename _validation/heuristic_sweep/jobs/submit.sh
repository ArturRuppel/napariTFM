#!/bin/bash
# Submit an array job sized to the number of scenes. Run ON MAESTRO after:
#   source $STAGE/env.sh
# Usage: bash submit.sh {cache|sweep} [max_concurrent]   (default 4, per lab GPU norm)
set -euo pipefail
: "${STAGE:?source env.sh first}" "${CODE:?}"
STAGEN="${1:?usage: submit.sh cache|sweep [max_concurrent]}"
CONCURRENCY="${2:-${CONCURRENCY:-4}}"
case "$STAGEN" in
    cache) SB=build_cache.sbatch ;;
    sweep) SB=sweep.sbatch ;;
    *) echo "unknown stage '$STAGEN' (cache|sweep)"; exit 1 ;;
esac
cd "$CODE"
# Flat count over every condition x scene (one array task each).
N=$(cd "$STAGE/scenes" && ls -d */* 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "no scenes under $STAGE/scenes"; exit 1; }
echo "submitting $STAGEN as array 0-$((N-1))%${CONCURRENCY} over $N (condition x scene) tasks (max ${CONCURRENCY} concurrent)"
sbatch --array="0-$((N-1))%${CONCURRENCY}" "jobs/$SB"

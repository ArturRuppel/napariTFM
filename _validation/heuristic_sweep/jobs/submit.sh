#!/bin/bash
# Submit an array job sized to the number of scenes. Run ON MAESTRO after:
#   source $STAGE/env.sh
# Usage: bash submit.sh {cache|sweep} [max_concurrent]   (default 4, per lab GPU norm)
set -euo pipefail
: "${STAGE:?source env.sh first}" "${CODE:?}" "${CONDITION:?}"
STAGEN="${1:?usage: submit.sh {cache|sweep} [max_concurrent]}"
CONCURRENCY="${2:-${CONCURRENCY:-4}}"
case "$STAGEN" in
    cache) SB=build_cache.sbatch ;;
    sweep) SB=sweep.sbatch ;;
    *) echo "unknown stage '$STAGEN' (cache|sweep)"; exit 1 ;;
esac
cd "$CODE"
N=$(ls -1 "$STAGE/scenes/$CONDITION" | wc -l)
[ "$N" -gt 0 ] || { echo "no scenes in $STAGE/scenes/$CONDITION"; exit 1; }
echo "submitting $STAGEN as array 0-$((N-1))%${CONCURRENCY} over $N scenes (max ${CONCURRENCY} concurrent)"
sbatch --array="0-$((N-1))%${CONCURRENCY}" "jobs/$SB"

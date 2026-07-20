#!/bin/bash
# Submit the sweep as a two-stage, cross-partition pipeline. Run ON MAESTRO after:
#   source $STAGE/env.sh
#
# Usage:
#   bash submit.sh cache    [gpu_conc]                 # stage 1 only (gpu partition)
#   bash submit.sh sweep    [cpu_conc] [--after JOBID] # stage 2 only (common partition)
#   bash submit.sh pipeline [gpu_conc] [cpu_conc]      # both, sweep chained to cache
#
# `pipeline` is the normal launch: it submits the GPU cache, then the CPU sweep
# with --dependency=aftercorr:<cache_jobid>, so each scene's sweep starts the
# instant that scene's cache completes -- no barrier between the stages.
# Defaults: gpu_conc=4 (partition is contended), cpu_conc=64 (common has cores).
set -euo pipefail
: "${STAGE:?source env.sh first}" "${CODE:?}"
cd "$CODE"

n_scenes() {
    # SCENE_GLOB (default */*) restricts the array to one condition, e.g.
    # SCENE_GLOB="cell_s6j1/*" runs only the cell scenes. Must match the sbatch glob.
    local n; n=$(cd "$STAGE/scenes" && ls -d ${SCENE_GLOB:-*/*} 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "no scenes under $STAGE/scenes (glob ${SCENE_GLOB:-*/*})" >&2; exit 1; }
    echo "$n"
}

submit_cache() {   # $1=concurrency ; echoes the job id
    local conc="${1:-4}" n; n=$(n_scenes)
    echo "cache: array 0-$((n-1))%${conc} on gpu over $n scenes (all methods)" >&2
    sbatch --parsable --array="0-$((n-1))%${conc}" jobs/build_cache.sbatch
}

submit_sweep() {   # $1=concurrency, $2=optional dep jobid ; echoes the job id
    local conc="${1:-64}" dep="${2:-}" n dopt=(); n=$(n_scenes)
    [ -n "$dep" ] && dopt=(--dependency="aftercorr:${dep}")
    echo "sweep: array 0-$((n-1))%${conc} on common over $n scenes${dep:+ (aftercorr:$dep)}" >&2
    sbatch --parsable --array="0-$((n-1))%${conc}" "${dopt[@]}" jobs/sweep_cpu.sbatch
}

case "${1:?usage: submit.sh cache|sweep|pipeline ...}" in
    cache)
        JID=$(submit_cache "${2:-4}"); echo "submitted cache job $JID" ;;
    sweep)
        DEP=""; [ "${2:-}" = "--after" ] && DEP="${3:?--after needs a jobid}"
        [ "${3:-}" = "--after" ] && DEP="${4:?--after needs a jobid}"
        CONC="${2:-64}"; [ "$CONC" = "--after" ] && CONC=64
        JID=$(submit_sweep "$CONC" "$DEP"); echo "submitted sweep job $JID" ;;
    pipeline)
        GPU_CONC="${2:-4}"; CPU_CONC="${3:-64}"
        CID=$(submit_cache "$GPU_CONC")
        echo "submitted cache job $CID"
        SID=$(submit_sweep "$CPU_CONC" "$CID")
        echo "submitted sweep job $SID  (starts per-scene as cache $CID lands)"
        echo
        echo "watch:   squeue -u \$USER"
        echo "results: $STAGE/results/sweep_<condition>_<scene>.csv  (trickle in)" ;;
    *)
        echo "unknown '$1' (cache|sweep|pipeline)"; exit 1 ;;
esac

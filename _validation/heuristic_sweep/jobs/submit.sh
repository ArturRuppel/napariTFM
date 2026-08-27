#!/bin/bash
# Submit the sweep as a two-stage, cross-partition pipeline. Run ON MAESTRO after:
#   source $STAGE/env.sh
#
# Usage:
#   bash submit.sh cache    [gpu_conc]                  # stage 1 only (displacement)
#   bash submit.sh force    [conc] [--after JOBID]      # stage 2 only (oracle force)
#   bash submit.sh pipeline [gpu_conc] [force_conc]     # both, force chained to cache
#
# `pipeline` is the normal launch: it submits the displacement cache, then the
# GT-tuned oracle force cache with --dependency=aftercorr:<cache_jobid>, so each
# scene's force stage starts the instant that scene's displacement cache
# completes -- no barrier between the stages.
# Defaults: gpu_conc=4 (partition is contended), force_conc=4.
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

submit_force() {   # $1=concurrency, $2=optional dep jobid ; echoes the job id.
    local conc="${1:-4}" dep="${2:-}" n dopt=(); n=$(n_scenes)
    [ -n "$dep" ] && dopt=(--dependency="aftercorr:${dep}")
    echo "force: array 0-$((n-1))%${conc} on $(awk -F= '/^#SBATCH --partition=/{print $2}' jobs/force_cache.sbatch) over $n scenes (oracle FTTC+L2 & FISTA+L1)${dep:+ (aftercorr:$dep)}" >&2
    sbatch --parsable --array="0-$((n-1))%${conc}" "${dopt[@]}" jobs/force_cache.sbatch
}

case "${1:?usage: submit.sh cache|force|pipeline ...}" in
    cache)
        JID=$(submit_cache "${2:-4}"); echo "submitted cache job $JID" ;;
    force)
        DEP=""; [ "${2:-}" = "--after" ] && DEP="${3:?--after needs a jobid}"
        [ "${3:-}" = "--after" ] && DEP="${4:?--after needs a jobid}"
        CONC="${2:-4}"; [ "$CONC" = "--after" ] && CONC=4
        JID=$(submit_force "$CONC" "$DEP"); echo "submitted force job $JID" ;;
    pipeline)
        GPU_CONC="${2:-4}"; FORCE_CONC="${3:-4}"
        CID=$(submit_cache "$GPU_CONC")
        echo "submitted cache job $CID"
        FID=$(submit_force "$FORCE_CONC" "$CID")
        echo "submitted force job $FID  (starts per-scene as cache $CID lands)"
        echo
        echo "watch:  squeue -u \$USER"
        echo "then:   python render_cache.py --stage \$STAGE --workers N   # cards + index.csv"
        echo "        python analyze.py --stage \$STAGE --outdir ../../docs/images" ;;
    *)
        echo "unknown '$1' (cache|force|pipeline)"; exit 1 ;;
esac

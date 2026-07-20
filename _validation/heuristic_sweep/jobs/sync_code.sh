#!/bin/bash
# Stage tracked code onto /helix so Maestro can import it. /helix is a local
# CIFS mount here, so this is a plain local rsync -- no ssh. Run from local.
# Data dirs (scenes/cache/results/logs) and env.sh are left untouched.
set -euo pipefail
REPO="/home/aruppel/Projects/napariTFM"
STAGE="${STAGE:-/helix/projects/Pomice2/Artur/tfm_heuristic}"
CODE="$STAGE/code"
mkdir -p "$CODE"

# sweep scripts (flatten into $CODE so `import sweep_config` works)
rsync -a --delete-excluded \
    --include='*.py' --include='jobs/' --include='jobs/*' --include='README.md' \
    --exclude='*' \
    "$REPO/_validation/heuristic_sweep/" "$CODE/"

# napariTFM backend package (only what the sweep imports; keep it lean)
rsync -a --delete \
    --include='__init__.py' --include='backend/' --include='backend/**' \
    --exclude='*' \
    "$REPO/napariTFM/" "$CODE/napariTFM/"

echo "synced -> $CODE"
echo "note: ensure the Maestro venv (\$VENV) has napariTFM.backend's deps installed."

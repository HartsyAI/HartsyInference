#!/usr/bin/env bash
# Builds the reference venv and fetches the pinned upstream files. Idempotent.
set -euo pipefail
ROOT="${DSV41_REF_ROOT:-$HOME/dsv41-ref}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT"
[ -x "$ROOT/.venv/bin/python" ] || python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --quiet -r "$HERE/requirements.txt" \
  --extra-index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$ROOT/.venv/bin/python" "$HERE/fetch_upstream.py" --dest "$ROOT/upstream"
echo "ready: $ROOT/.venv/bin/python $HERE/run_small_config.py"

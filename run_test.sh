#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/d/Anaconda/python.exe" ]]; then
  PYTHON_BIN="/d/Anaconda/python.exe"
else
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$ROOT_DIR/main.py" "$@"

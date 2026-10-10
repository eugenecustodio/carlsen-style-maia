#!/bin/sh
# PyChess / any UCI GUI: point "New engine" at this file (protocol UCI).
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/../env/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$DIR/engine/carlsen_uci.py"

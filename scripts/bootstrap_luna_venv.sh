#!/usr/bin/env bash
set -euo pipefail
PY="$(ls -d /home/ai/.local/share/uv/python/cpython-3.11*/bin/python3 2>/dev/null | head -1)"
echo "using $PY"
"$PY" --version
cd /home/ai/regenes/client
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q "websockets==12.0" certifi "neat-python==1.1.0"
.venv/bin/python -c "import neat,websockets; print('neat', neat.__version__, 'ws', websockets.__version__)"
chmod +x scripts/*.sh
scripts/start_luna.sh
sleep 2
echo "=== procs ==="
ps aux | grep -E "host.py|host_hyper|host_grn" | grep -v grep || true
echo "=== native log ==="
tail -30 logs/native.log || true
echo "=== hyper log ==="
tail -30 logs/hyper.log || true
echo "=== grn log ==="
tail -30 logs/grn.log || true
sleep 4
curl -sS http://127.0.0.1:8081/
echo

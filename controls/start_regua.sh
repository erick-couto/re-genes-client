#!/usr/bin/env bash
# Uma instância da régua de cheiro. NÃO entra no start_luna.sh dos executores.
# Controle, não participante. Ver README.md e o card T7.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
WS="${REGENES_WS:-ws://127.0.0.1:8081}"
N="${N_REGUA:-1}"
LOG="${ROOT}/logs"
mkdir -p "$LOG"
if [[ ! -x "$PY" ]]; then
  echo "faltando venv em $PY" >&2
  exit 1
fi
# pkill: colchetes para não casar com este próprio shell.
pkill -f "client_regua_scen[t].py" 2>/dev/null || true
sleep 0.3
export REGENES_OPERATOR="${REGENES_OPERATOR:-regua}"
export REGENES_SERVER="$WS"
cd "$ROOT/controls"
nohup "$PY" -u client_regua_scent.py "$N" "$WS" >>"$LOG/regua.log" 2>&1 &
echo "regua pid $!  N=$N  $WS  (controle, nao-participante)"
echo "log: $LOG/regua.log"

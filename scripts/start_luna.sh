#!/usr/bin/env bash
# Sobe os dois executores Fase 2 contra o mundo local.
# NÃO sobe legacy/ (fitness explícita, card #10).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
WS="${REGENES_WS:-ws://127.0.0.1:8081}"
N_NATIVE="${N_NATIVE:-20}"
N_HYPER="${N_HYPER:-20}"
LOG="${ROOT}/logs"
mkdir -p "$LOG"

if [[ ! -x "$PY" ]]; then
  echo "faltando venv em $PY" >&2
  exit 1
fi

# pkill: colchetes pra não casar com este próprio shell (armadilha da CLAUDE.md).
pkill -f "client_native/hos[t].py" 2>/dev/null || true
pkill -f "client_hyperneat/host_hype[r].py" 2>/dev/null || true
sleep 0.3

export REGENES_OPERATOR="${REGENES_OPERATOR:-luna}"
cd "$ROOT/client_native"
nohup "$PY" -u host.py "$N_NATIVE" "$WS" >>"$LOG/native.log" 2>&1 &
echo "native pid $!  N=$N_NATIVE  $WS"

cd "$ROOT/client_hyperneat"
nohup "$PY" -u host_hyper.py "$N_HYPER" "$WS" >>"$LOG/hyper.log" 2>&1 &
echo "hyper  pid $!  N=$N_HYPER  $WS"
echo "logs: $LOG/{native,hyper}.log"

#!/usr/bin/env bash
# Sobe os três executores Fase 2 contra o mundo local.
# NÃO sobe legacy/ (fitness explícita, card #10).
#
# NÃO empilha. O stop antigo procurava `client_native/host.py` no argv; o start
# roda `python -u host.py` depois do cd — pkill era no-op e o segundo start
# gerava 80 conexões no teto de 50 (Luna 24/08). Sempre paramos pelo stop_luna.sh
# (único padrão) e ABORTAMOS se ainda houver processo. Não invente pkill.
#
# Padrões com colchetes: host.py / host_hyper.py / host_grn.py são argv distintos.
# `hos[t].py` NÃO casa host_grn.py nem host_hyper.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
WS="${REGENES_WS:-ws://127.0.0.1:8081}"
N_NATIVE="${N_NATIVE:-20}"
N_HYPER="${N_HYPER:-20}"
N_GRN="${N_GRN:-20}"
LOG="${ROOT}/logs"
mkdir -p "$LOG"

if [[ ! -x "$PY" ]]; then
  echo "faltando venv em $PY" >&2
  exit 1
fi

bash "$ROOT/scripts/stop_luna.sh"
sleep 0.5

_count() { pgrep -c -f "$1" 2>/dev/null || true; }

set +e
ainda_n=$(_count 'python -u hos[t].py')
ainda_h=$(_count 'python -u host_hype[r].py')
ainda_g=$(_count 'python -u host_gr[n].py')
set -e
ainda_n=${ainda_n:-0}
ainda_h=${ainda_h:-0}
ainda_g=${ainda_g:-0}
if [[ "$ainda_n" -gt 0 || "$ainda_h" -gt 0 || "$ainda_g" -gt 0 ]]; then
  echo "clientes ainda vivos (native=$ainda_n hyper=$ainda_h grn=$ainda_g) — kill -9" >&2
  pkill -9 -f "python -u hos[t].py" 2>/dev/null || true
  pkill -9 -f "python -u host_hype[r].py" 2>/dev/null || true
  pkill -9 -f "python -u host_gr[n].py" 2>/dev/null || true
  sleep 0.4
fi
set +e
ainda_n=$(_count 'python -u hos[t].py')
ainda_h=$(_count 'python -u host_hype[r].py')
ainda_g=$(_count 'python -u host_gr[n].py')
set -e
ainda_n=${ainda_n:-0}
ainda_h=${ainda_h:-0}
ainda_g=${ainda_g:-0}
if [[ "$ainda_n" -gt 0 || "$ainda_h" -gt 0 || "$ainda_g" -gt 0 ]]; then
  echo "ABORT: nao vou empilhar. native=$ainda_n hyper=$ainda_h grn=$ainda_g ainda no ar." >&2
  exit 1
fi

export REGENES_OPERATOR="${REGENES_OPERATOR:-luna}"
cd "$ROOT/client_native"
nohup "$PY" -u host.py "$N_NATIVE" "$WS" >>"$LOG/native.log" 2>&1 &
echo "native pid $!  N=$N_NATIVE  $WS"

cd "$ROOT/client_hyperneat"
nohup "$PY" -u host_hyper.py "$N_HYPER" "$WS" >>"$LOG/hyper.log" 2>&1 &
echo "hyper  pid $!  N=$N_HYPER  $WS"

cd "$ROOT/client_grn"
nohup "$PY" -u host_grn.py "$N_GRN" "$WS" >>"$LOG/grn.log" 2>&1 &
echo "grn    pid $!  N=$N_GRN  $WS"

sleep 0.3
set +e
nn=$(_count 'python -u hos[t].py')
nh=$(_count 'python -u host_hype[r].py')
ng=$(_count 'python -u host_gr[n].py')
set -e
nn=${nn:-0}
nh=${nh:-0}
ng=${ng:-0}
echo "procs agora: native=$nn hyper=$nh grn=$ng  (esperado 1 1 1)"
if [[ "$nn" -ne 1 || "$nh" -ne 1 || "$ng" -ne 1 ]]; then
  echo "ALERTA: nao e 1+1+1. Nao suba start de novo — rode stop_luna.sh e investigue." >&2
  exit 1
fi
echo "logs: $LOG/{native,hyper,grn}.log"

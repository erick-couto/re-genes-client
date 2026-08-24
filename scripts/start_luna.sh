#!/usr/bin/env bash
# Sobe os dois executores Fase 2 contra o mundo local.
# NÃO sobe legacy/ (fitness explícita, card #10).
#
# NÃO empilha. O stop antigo procurava `client_native/host.py` no argv; o start
# roda `python -u host.py` depois do cd — pkill era no-op e o segundo start
# gerava 80 conexões no teto de 50 (Luna 24/08). Sempre paramos pelo stop_luna.sh
# (único padrão) e ABORTAMOS se ainda houver processo. Não invente pkill.
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

bash "$ROOT/scripts/stop_luna.sh"
sleep 0.5

set +e
ainda_n=$(pgrep -c -f 'python -u hos[t].py' 2>/dev/null)
ainda_h=$(pgrep -c -f 'python -u host_hype[r].py' 2>/dev/null)
set -e
ainda_n=${ainda_n:-0}
ainda_h=${ainda_h:-0}
if [[ "$ainda_n" -gt 0 || "$ainda_h" -gt 0 ]]; then
  echo "clientes ainda vivos (native=$ainda_n hyper=$ainda_h) — kill -9" >&2
  pkill -9 -f "python -u hos[t].py" 2>/dev/null || true
  pkill -9 -f "python -u host_hype[r].py" 2>/dev/null || true
  sleep 0.4
fi
set +e
ainda_n=$(pgrep -c -f 'python -u hos[t].py' 2>/dev/null)
ainda_h=$(pgrep -c -f 'python -u host_hype[r].py' 2>/dev/null)
set -e
ainda_n=${ainda_n:-0}
ainda_h=${ainda_h:-0}
if [[ "$ainda_n" -gt 0 || "$ainda_h" -gt 0 ]]; then
  echo "ABORT: nao vou empilhar. native=$ainda_n hyper=$ainda_h ainda no ar." >&2
  exit 1
fi

export REGENES_OPERATOR="${REGENES_OPERATOR:-luna}"
cd "$ROOT/client_native"
nohup "$PY" -u host.py "$N_NATIVE" "$WS" >>"$LOG/native.log" 2>&1 &
echo "native pid $!  N=$N_NATIVE  $WS"

cd "$ROOT/client_hyperneat"
nohup "$PY" -u host_hyper.py "$N_HYPER" "$WS" >>"$LOG/hyper.log" 2>&1 &
echo "hyper  pid $!  N=$N_HYPER  $WS"

sleep 0.3
set +e
nn=$(pgrep -c -f 'python -u hos[t].py' 2>/dev/null)
nh=$(pgrep -c -f 'python -u host_hype[r].py' 2>/dev/null)
set -e
nn=${nn:-0}
nh=${nh:-0}
echo "procs agora: native=$nn hyper=$nh  (esperado 1 e 1)"
if [[ "$nn" -ne 1 || "$nh" -ne 1 ]]; then
  echo "ALERTA: nao e 1+1. Nao suba start de novo — rode stop_luna.sh e investigue." >&2
  exit 1
fi
echo "logs: $LOG/{native,hyper}.log"

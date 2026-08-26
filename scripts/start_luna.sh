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
N_NATIVE="${N_NATIVE:-15}"
N_HYPER="${N_HYPER:-15}"
N_GRN="${N_GRN:-15}"
# #54 passo 1 — EM QUANTOS PROCESSOS cada arquitetura se divide.
#
# Um processo Python atende suas amebas SEQUENCIALMENTE (o GIL não deixa outra coisa),
# então o lote inteiro espera a mais lenta. MEDIDO no mundo em 26/08/2026, com 1 processo
# de 15 amebas por arquitetura:
#
#   GRN        média 10,35 ms   |  >16 ms em  4,2% das respostas
#   HyperNEAT  média 14,97 ms   |  >16 ms em 11,2%
#   Native     média 11,32 ms   |  >16 ms em  4,5%
#
# Os 16 ms são o tick que um mundo a 60 Hz teria. Com essa distribuição, uma barreira de
# prazo curto cortaria o HyperNEAT 2,5x mais que os outros — seleção por custo de
# inferência, que o #54 chama de cobrança contrabandeada (o cérebro JÁ é cobrado de forma
# declarada em §15.3 e §21).
#
# Dividir o lote ataca a causa em vez de mascarar com tolerância: a Luna tem 14 núcleos
# e o mundo usa 0,27 de um. SPLIT=5 dá 3 amebas por processo e a fila de cada uma encolhe
# na mesma proporção. SPLIT=1 reproduz o comportamento anterior, sem surpresa.
SPLIT="${SPLIT:-1}"
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

# Reparte N amebas em SPLIT processos. O resto vai para os primeiros, então a soma
# fecha EXATAMENTE em N — sem isso, dividir 15 por 4 perderia amebas em silêncio.
_sobe() {   # $1=dir  $2=script  $3=N  $4=rótulo
  local dir="$1" script="$2" total="$3" rot="$4"
  local base=$(( total / SPLIT )) resto=$(( total % SPLIT )) i n soma=0
  cd "$ROOT/$dir"
  for (( i = 0; i < SPLIT; i++ )); do
    n=$base
    (( i < resto )) && n=$(( n + 1 ))
    [[ "$n" -eq 0 ]] && continue
    nohup "$PY" -u "$script" "$n" "$WS" >>"$LOG/$rot.log" 2>&1 &
    echo "$rot pid $!  N=$n  $WS"
    soma=$(( soma + n ))
  done
  if [[ "$soma" -ne "$total" ]]; then
    echo "ABORT: $rot subiu $soma amebas, esperado $total" >&2
    exit 1
  fi
}

_sobe client_native    host.py       "$N_NATIVE" native
_sobe client_hyperneat host_hyper.py "$N_HYPER"  hyper
_sobe client_grn       host_grn.py   "$N_GRN"    grn

sleep 0.3
set +e
nn=$(_count 'python -u hos[t].py')
nh=$(_count 'python -u host_hype[r].py')
ng=$(_count 'python -u host_gr[n].py')
set -e
nn=${nn:-0}
nh=${nh:-0}
ng=${ng:-0}
echo "procs agora: native=$nn hyper=$nh grn=$ng  (esperado $SPLIT $SPLIT $SPLIT)"
if [[ "$nn" -ne "$SPLIT" || "$nh" -ne "$SPLIT" || "$ng" -ne "$SPLIT" ]]; then
  echo "ALERTA: nao e $SPLIT+$SPLIT+$SPLIT. Nao suba start de novo — rode stop_luna.sh e investigue." >&2
  exit 1
fi
echo "amebas: native=$N_NATIVE hyper=$N_HYPER grn=$N_GRN em $SPLIT processo(s) cada"
echo "logs: $LOG/{native,hyper,grn}.log"

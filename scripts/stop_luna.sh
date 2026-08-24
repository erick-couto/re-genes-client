#!/usr/bin/env bash
# Colchetes no padrão: pkill -f com o nome do script na mesma linha se mata.
# start_luna.sh faz `cd …/client_native && python -u host.py` — o argv NÃO contém
# o caminho client_native/host.py. O padrão antigo não matava ninguém e o próximo
# start empilhava outro 20+20 (Luna 24/08: 50 vivos no teto + ~30 na incubadora).
set -euo pipefail
pkill -f "python -u hos[t].py" 2>/dev/null || true
pkill -f "python -u host_hype[r].py" 2>/dev/null || true
echo "clientes luna parados"

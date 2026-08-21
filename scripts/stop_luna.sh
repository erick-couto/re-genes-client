#!/usr/bin/env bash
# Colchetes no padrão: pkill -f com o nome do script na mesma linha se mata.
set -euo pipefail
pkill -f "client_native/hos[t].py" 2>/dev/null || true
pkill -f "client_hyperneat/host_hype[r].py" 2>/dev/null || true
echo "clientes luna parados"

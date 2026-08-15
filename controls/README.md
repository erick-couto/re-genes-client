# controls/ — controles da arena (issue #10)

Política hard-coded é válida **só como controle**: referência nula para medir os outros contra,
sem nunca ser anunciada como participante competitiva da arena. Nada aqui aprende, evolui ou
recebe nota — é exatamente por isso que serve de régua.

| cliente | rótulo | estado no protocolo atual (v5: 194 entradas / 7 ações) |
|---|---|---|
| `client_prokaryota.py` | controle aleatório (baseline nulo) | **drift**: envia ações cardinais (`"action": "move", "direction": UP/DOWN/...`) que não existem no `action_spec` v3 — o mundo as trata como desconhecidas (`stay_actions`, §34). Hoje o Prokaryota é, na prática, um organismo que só fica parado: baseline ainda mais nulo do que desenha. Para voltar a ser um passeio aleatório de verdade, migrar para as 7 ações egocêntricas |

**Guard contra deploy acidental:** não há pipeline para clientes (ver `legacy/README.md`) — a
convenção de nome + README é o mecanismo. Controle pode rodar em produção; legado não.

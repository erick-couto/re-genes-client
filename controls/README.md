# controls/ — controles da arena (issue #10)

Política hard-coded é válida **só como controle**: referência nula para medir os outros contra,
sem nunca ser anunciada como participante competitiva da arena. Nada aqui aprende, evolui ou
recebe nota — é exatamente por isso que serve de régua.

| cliente | rótulo | estado no protocolo atual (v7: 163 entradas / 7 ações) |
|---|---|---|
| `client_regua_scent.py` | régua de quimiotaxia (T7) | **ativo**. Lê cone 4×31 + químico 3×9, anda no gradiente de cheiro da própria dieta inferida, contorna parede, pasta o que estiver embaixo. `wants_brain=0`, `self_learns=0`, species=`ReguaScent`. Não aprende, não entra em `brain_bank`, **não é participante**. N=1. Não sobe pelo `start_luna.sh`. |
| `client_prokaryota.py` | controle aleatório (baseline nulo) | **aposentado**. Drift: ações cardinais que o v7 trata como `stay`. Não usar. |

**Guard contra deploy acidental:** não há pipeline para clientes (ver `legacy/README.md`) — a
convenção de nome + README é o mecanismo. Controle pode rodar em produção; legado não.
A régua sobe só por `controls/start_regua.sh` (N=1), nunca pelo `scripts/start_luna.sh`.

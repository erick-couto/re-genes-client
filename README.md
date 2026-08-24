# 🧬 re-genes.is: Neural Client

> **Workspace da arena:** se você está em `D:\DESENVOLVIMENTO\regenes`, leia
> [`../AGENTS.md`](../AGENTS.md) primeiro. Protocolo **v7 / 163 entradas**
> (`host.py:40-41`). `legacy/` fora da Luna (#10). Deploy na Luna **não** é
> `git pull` — ver `LUNA_HANDOFF.md`.

> **Neural Modules (Brains) for the Re-Genes simulation.**

## Sobre
Este repositório contém os agentes autônomos (Amebas) que habitam o servidor **re-genes-world**.
O servidor descreve o mundo (protocolo auto-descritivo); o cliente traz a inteligência. Sem nota
externa: seleção é sobreviver + reproduzir no mundo.

## Estrutura (quarentena — issue #10)

| diretório | papel | pode rodar na arena sem-reward? |
|---|---|---|
| `client_native/` | **executor Fase 2** — NEAT nativo (genoma direto) | ✅ |
| `client_hyperneat/` | **executor Fase 2** — HyperNEAT (CPPN, codificação indireta) | ✅ |
| `regenes_agent.py` | SDK do Species Protocol (BaseAgent + loop) | — |
| `controls/` | políticas hard-coded — válidas **só como controle** | ✅ (como referência) |
| `legacy/` | quarentena: fitness explícita, valência fabricada, Lamarckismo e drift de protocolo | ❌ |

**Nada do que está em `legacy/` deve ser anunciado como participante da arena nem ligado contra
`wss://re-genes.is`.** Ver `legacy/README.md` e `controls/README.md` para o inventário completo
(o que cada um viola, e por quê).

## Versão de protocolo e shape por cliente (v7 atual: 163 entradas / 7 ações)

| cliente | entradas que monta | ações | handshake |
|---|---|---|---|
| `client_native` | **163** = 12 slots de escalares (bias acrescentado localmente; o `stomach_size` que o mundo envia vira denominador da normalização interoceptiva, §43/§50/§51) + 4 canais × 31 do cone (obstáculos, corpo, perigo, comida — borrados pela acuidade) + 3 campos × 9 químico por contato (cheiro-planta, cheiro-carne, sangue, §52) (`host.py:encode`, `host.py:40-41`) | 7 egocêntricas | `species=Native_NEAT`, `wants_brain=1`, `self_learns=0` |
| `client_hyperneat` | **163** via substrato expresso pelo CPPN (163 entradas + 16 ocultos + 7 saídas = nós fixos, `substrate.py:75`; √n por sinapse, `substrate.py:147-169`) | 7 egocêntricas | `species=HyperNEAT`, `wants_brain=1`, `self_learns=0` |
| `controls/client_prokaryota` | ignora a visão | cardinais (drift — o mundo trata como desconhecidas) | `species=Prokaryota` |
| `legacy/client_neat` | 104 montadas vs config 161 (drift) | 7 no SDK / cardinais no standalone | `species=NEAT_Evo` |
| `legacy/client_es` | 159 (v6, sem sangue/ingested) | 7 | `species=ES_v1` |
| `legacy/client_memoriam` | Q-table por fenótipo | 9 ações cardinais (v2) | `species=Memoriam`, `self_learns=1` (Lamarckiano) |
| `legacy/client.py` | nenhuma | cardinais (protocolo morto) | — |

## Como Rodar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute um executor Fase 2 (ex: nativo, N amebas contra o servidor):
   ```bash
   python client_native/host.py 8 wss://re-genes.is
   ```

## Arquitetura
Os clientes operam em modo **Reativo**:
1. Conectam ao WebSocket do servidor (`/ws/join`, com handshake de espécie).
2. Aguardam o sinal de `TICK`.
3. Processam a observação do mundo (4 canais de cone egocêntrico + campos químicos por contato + escalares, protocolo v7).
4. Enviam a decisão (uma das 7 ações egocêntricas).

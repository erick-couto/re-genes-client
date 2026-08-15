# 🧬 re-genes.is: Neural Client

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

## Versão de protocolo e shape por cliente (v5 atual: 194 entradas / 7 ações)

| cliente | entradas que monta | ações | handshake |
|---|---|---|---|
| `client_native` | **194** = bias + energy + stomach + ingested + marca-passo(sin,cos) + damage + impact (normalizados pelo estômago) + 6 canais × 31 do cone, borrados pela acuidade (`host.py:encode`) | 7 egocêntricas | `species=Native_NEAT`, `wants_brain=1`, `self_learns=0` |
| `client_hyperneat` | **194** via substrato expresso pelo CPPN (194+16+7 nós fixos) | 7 egocêntricas | `species=HyperNEAT`, `wants_brain=1`, `self_learns=0` |
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
3. Processam a visão do mundo (cone egocêntrico de 6 canais).
4. Enviam a decisão (uma das 7 ações egocêntricas).

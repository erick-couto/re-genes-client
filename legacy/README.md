# legacy/ — quarentena (issue #10)

Separar e rotular, **não deletar**: nada aqui é apagado. Estes clientes estão fora da arena
sem-reward do re-genes. **Não subir contra `wss://re-genes.is`** — e nenhum deles deve ser
anunciado como participante da arena (a referência pública vive no re-genes-site, corrigida
no card #10).

**Por que a quarentena existe (a regra que governa):** o mundo molda a física; seleção é
sobreviver + reproduzir no mundo, sem nota externa. Tudo aqui viola isso de alguma forma —
fitness explícita fabricada pelo cliente, valência inventada fora do mundo, ou herança do
adquirido em vida — além de drift de protocolo (shape/ações que não existem mais).

**Guard contra deploy acidental:** os clientes NÃO têm mecanismo de deploy automático — não há
Jenkinsfile, Dockerfile nem compose neste repo (o Jenkins que existe é do re-genes-world e só
toca o mundo). Eles entram no ar manualmente (`python ... run.py N URL`). Sem pipeline, o guard
é esta convenção: **tudo que não está na raiz nem é Fase 2 vive em `legacy/` ou `controls/`**,
e "rodar um legado em produção" exige atravessar um diretório que diz NÃO. Quem rodar um
legado, roda sabendo.

## Inventário (verificado no fonte em 15/08/2026)

| cliente | rótulo | entradas que monta / ações | protocolo atual (v7: 163/7) | o que viola |
|---|---|---|---|---|
| `client_neat/` | legado com fitness explícita | **104** entradas montadas (bias+energy+stomach+endorfina + 4 canais × 25 células) contra config `num_inputs = 161` (`client_neat/config-feedforward:87`) | 163 entradas, 7 ações egocêntricas | fitness explícita em `on_death` (pesos FOOD/ENERGY/SURVIVAL/EXPLORE, `neat_agent.py`); **endorfina fabricada** (+0,3 por mover, +5,0 por célula nova, +100 por ganho de energia — `neat_agent.py:84-89`); comentários de direção de comportamento no legado standalone ("REHAB: Massive reward to make eating the primary goal", `client_neat.py:374`); o standalone usa ações CARDINAIS (UP/DOWN/LEFT/RIGHT/STAY, `client_neat.py:294`) que o mundo de hoje trata como desconhecidas |
| `client_es/` | legado com fitness explícita | **159** entradas (5 canais × 31 + energy/stomach/marca-passo — falta o canal de sangue e o escalar `ingested`) | 163 entradas, 7 ações | fitness explícita por perturbação (ES reporta fitness, `es_agent.py`); shape antigo (v6, pré-§43) |
| `client_memoriam/` | **controle/baseline Lamarckiano** (não apenas "legado") | **9 ações cardinais** (`client_memoriam.py:57` — espaço de ações v2: UP/DOWN/LEFT/RIGHT/STAY/ATK_*) | 7 ações egocêntricas | **herança do adquirido**: Q-table por fenótipo persiste ENTRE VIDAS (`memoriam_agent.py:6-7`, `self_learns=True`) — Lamarckismo, explicitamente fora do modelo Baldwiniano canônico (issue #26). As tabelas persistidas (`qtable_memoriam_*.json`) são o artefato da herança |
| `client.py` | legado (cliente-raiz histórico) | ações cardinais aleatórias (`"action": "move", "direction": ...`) e leitura de STATE como se fosse TICK | protocolo v7 (ele fala v5) | política aleatória de protocolo antigo: não lê WELCOME/TICK corretamente, none das ações existe no `action_spec` v3 |

**Sobre o Memoriam:** o rótulo correto, definido na auditoria do card #10, é
**"controle/baseline Lamarckiano"**: ele vale como referência comparativa EXATAMENTE porque
viola o cânone (herança do aprendido em vida), e é assim que deve ser citado — nunca como
participante da arena sem-reward.

**O que o mundo faz se um legado conectar:** nada o impede mecanicamente — o handshake aceita
qualquer `species`, a observação sai do mundo para o cliente independentemente do shape que o
cliente monta, e o isolamento reprodutivo é pelo `brain_name` declarado (legado só cruza com o
próprio rótulo). Comandos fora do `action_spec` v3 viram `stay` (contados em `stay_actions`,
§34 do mundo). Ou seja: um legado em produção não quebra o mundo, mas ocupa slot, come comida e
polui a telemetria — por isso fica quarentenado.

**Como reabilitar (se um dia):** reescrever para o protocolo v7 (163 entradas, 7 ações
egocêntricas, `regenes_agent.py` na raiz), remover toda fitness/valência fabricada e toda
herança entre vidas, e passar pelo crivo do card antes de voltar à arena.

"""
host.py — Executor de cérebro-no-genoma (Fase 2, lado CLIENTE).

Conecta N amebas com wants_brain=1. Para cada uma:
  1. recebe do mundo a SEMENTE de cérebro (blob do pai/ancestral, ou None p/ primordial);
  2. monta o cérebro NEAT (herda + muta, ou cria novo aleatório) — a herança;
  3. reporta o cérebro final pro mundo guardar (pra herdar/arquivar depois);
  4. roda o forward pass a cada tick (execução distribuída, no cliente) e envia a ação;
  5. ao morrer, reconecta -> nova ameba, nova semente vinda do mundo.

NÃO mantém população nem função de fitness. Quem seleciona é o MUNDO (sobreviver e
reproduzir). O cérebro é do genoma (do mundo); o cliente só executa.

Uso:
    python host.py [N] [ws_base]
      N       -> quantas amebas nativas (default 8)
      ws_base -> ex.: ws://127.0.0.1:8000 (default). Produção (wss) precisa de SSL (TODO).
"""
import asyncio
import json
import math
import os
import random
import ssl
import sys
import time

import websockets
import neat_brain as nb
import cone_psf                      # R-BLUR: PSF na geometria do cone (compartilhado c/ o hyper)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000"
OP = os.getenv("REGENES_OPERATOR", "")  # dono da linhagem (carimbo na genealogia)
# §46 (R-SHAPE, card #38): o contrato DECLARADO no join — o mundo valida contra o
# /protocol dele (passo 1: avisa; passo 2: recusa com close 4001). n_obs = o que o
# encode() abaixo monta (8 escalares + 6×31 do cone); n_actions = len(ACTIONS).
# Os três valores andam juntos com o encode/ACTIONS: se o shape mudar, muda aqui.
PROTOCOL_VERSION = 6
N_OBS = 198
N_ACTIONS = 7
URL = (BASE.rstrip("/") + "/ws/join?species=Native_NEAT&paradigm=neuroevolution_topology"
       "&wants_brain=1&self_learns=0"
       f"&protocol_version={PROTOCOL_VERSION}&n_obs={N_OBS}&n_actions={N_ACTIONS}"
       + (f"&operator={OP}" if OP else ""))

# TELEMETRIA LOCAL de complexidade do cérebro (a produção só guarda sumários; isto dá a curva
# na hora, sem depender de deploy do mundo). 1 linha por nascimento. Append síncrono é seguro no
# asyncio single-thread (sem await no meio). Desligar com REGENES_TELEMETRY=0.
_TELEMETRY = os.path.join(os.path.dirname(__file__), "native_telemetry_v2.csv")
_TELEMETRY_ON = os.getenv("REGENES_TELEMETRY", "1") != "0"


def _telemetry(idx: int, origin: str, nodes: int, conns: int,
               fnodes: int, fconns: int, genes: int, acuity: float) -> None:
    # v2 (07/2026): fnodes/fconns = CÉREBRO REAL (sub-rede funcional, alcança as saídas);
    # genes = tamanho total do genoma (o que o §21 cobra). A série v1 (nodes/conns = genoma)
    # ficou em native_telemetry.csv — 148k linhas com header de 6 colunas; misturar formatos
    # no mesmo arquivo quebraria o DictReader, então a v2 nasce em arquivo novo.
    if not _TELEMETRY_ON:
        return
    try:
        new = not os.path.exists(_TELEMETRY)
        with open(_TELEMETRY, "a", encoding="ascii") as f:
            if new:
                f.write("unix_time,idx,origin,nodes,conns,fnodes,fconns,genes,acuity\n")
            f.write(f"{time.time():.0f},{idx},{origin},{nodes},{conns},"
                    f"{fnodes},{fconns},{genes},{acuity:.3f}\n")
    except OSError:
        pass  # telemetria nunca derruba o executor


def _ssl_ctx():
    """SSL só p/ wss. Tolera o MITM do Avast (VERIFY_X509_STRICT); REGENES_INSECURE_TLS=1 desliga tudo."""
    if not URL.startswith("wss"):
        return None
    ctx = ssl.create_default_context()
    if os.getenv("REGENES_INSECURE_TLS") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


SSL = _ssl_ctx()

# FÍSICA DA PERCEPÇÃO — acuidade ∝ capacidade neural (ver docs/FISICA_DA_PERCEPCAO.md no world).
# "Você enxerga na resolução que seu cérebro consegue processar." A visão é BORRADA por um
# desfoque gaussiano CONTÍNUO no cone, com largura sigma ∝ (1−A): cérebro pobre vê tudo smeared
# (só o borrão); afina LISO conforme a linhagem evolui — CADA conexão a mais deixa um tico mais
# nítido (gradiente SEM platô, pra a seleção conseguir catar; a versão em degraus travava a
# catraca — dentes a ~80 conexões um do outro). Determinístico (sem ruído aleatório -> auditável).
# Lei de embodiment (como cubo-quadrado/Kleiber), não currículo. Fixo no nascimento, client-side.
ACUITY_K = 120.0        # meia-saturação: A = C/(C+K). Baixo -> gradiente morde na faixa magra.
ACUITY_SIGMA_MAX = 6.0  # desfoque máximo (sigma no cone de 31 células) em A=0
# R5/#2 (08/2026): SEM fade semântico. O pred_w zerava os canais 2/3 (inimigo/perigo) pra
# A<0,40 — o executor EDITANDO a própria observação. O mundo descreve; quem interpreta é o
# cérebro. A única física da percepção é a PSF acima, aplicada igual a TODOS os canais.


def acuity_params(conns):
    """C (conexões) -> (PSF_do_cone, sigma, A). Fixo no nascimento.

    R-BLUR (§8 do Relatório Guardião): a PSF passou a viver na GEOMETRIA do cone
    (cone_psf.py), não no índice do buffer. A LEI não muda — A = C/(C+K),
    sigma = SIGMA_MAX*(1-A) —, muda a métrica de vizinhança."""
    A = conns / (conns + ACUITY_K)
    sigma = ACUITY_SIGMA_MAX * (1.0 - A)
    return (cone_psf.psf(sigma), sigma, A)


def _blur(row, P):
    """Aplica a PSF geométrica do cone (cone_psf). Substitui a convolução no índice serial,
    que misturava a extrema direita de uma fileira com a extrema esquerda da seguinte."""
    return cone_psf.blur(row, P)


# Encoding v3 EGOCÊNTRICO: 161 = bias + energy + stomach + ingested + MARCA-PASSO(sin,cos)
# + 5 canais x 31 do CONE, BORRADOS pela acuidade do cérebro (desfoque contínuo, igual em todos).
def encode(vision, energy, stomach, stomach_size, ingested, pace_sin, pace_cos, acuity,
           damage=0.0, impact=0.0,
           moved_self=0.0, moved_passive=0.0, contact_body=0.0, contact_wall=0.0):
    if not vision or len(vision) < 6 or len(vision[0]) < 31:
        return [0.0] * 198
    P = acuity[0]
    ss = stomach_size or 1.0
    # §26: damage/impact = FATO BRUTO interoceptivo (dano de mordida e impacto de colisão
    # sofridos neste tick), normalizados pelo PRÓPRIO estômago (egocêntrico, sinal positivo).
    # Não é valência: o mundo diz "aconteceu, nesta quantidade"; o que vale é do cérebro.
    # R4/#3 (08/2026): ingested idem — quanto o mundo reportou INGERIDO neste tick, no MESMO
    # idioma de normalização. Substitui a endorfina que o executor FABRICAVA (pico +100 por
    # delta de energia, decaimento, penalidade de fome): estado interno inventado, não fato.
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, min(1.0, ingested / ss),
           pace_sin, pace_cos, min(1.0, damage / ss), min(1.0, impact / ss),
           # §50/§51 (#43): quatro fatos que o mundo passou a entregar. Ja chegam
           # normalizados (bits e fracao sobre 4), entao NAO passam pelo estomago.
           #   moved_self/moved_passive: PROPRIOCEPCAO. A rede e feedforward, sem
           #     memoria — sem estes bits, "andei" e "esbarrei num corpo" tem o mesmo
           #     vetor de entrada, e ser deslocada por forca externa era invisivel.
           #   contact_body/contact_wall: PELE. Fracao das 4 ortogonais ocupadas,
           #     360 graus, independente do heading. O cone e OLHO e nao ve atras;
           #     estar cercada e exatamente quando a informacao esta fora do cone.
           moved_self, moved_passive, contact_body, contact_wall]
    # §23: 6º canal (sangue) entra como o cheiro — traço QUÍMICO, legível por qualquer cérebro.
    for ch in range(6):
        inp.extend(_blur(vision[ch], P))
    return inp

# índice -> comando de wire (bate com ACTION_SPEC do mundo, v3 egocêntrico: 7 ações)
ACTIONS = [
    {"action": "forward"},              # 0: anda pra frente (onde encara)
    {"action": "backward"},             # 1: recua (ré, sem virar)
    {"action": "turn", "dir": "left"},  # 2: gira à esquerda
    {"action": "turn", "dir": "right"}, # 3: gira à direita
    {"action": "stay"},                 # 4: fica
    {"action": "attack"},               # 5: morde a célula à frente
    {"action": "push"},                 # 6: empurra a célula à frente (sem dano; massa decide)
]


NULL_EPS = 0.05   # abaixo disto, a saída é ruído: o cérebro não disse nada


def decide(out):
    """Saídas da rede -> índice da ação. Três casos, e cada um tem uma razão física.

    1) SEM SINAL (tudo ~0) -> FICA. Nervo desconectado não dispara músculo: sem comando motor,
       o bicho não se mexe. Antes, um cérebro SEM conexões caía no argmax e ganhava "frente"
       DE GRAÇA — só porque frente é o índice 0. Um passeio em linha reta de presente, dado
       pela ORDEM em que as ações foram listadas. Era o mesmo viés-índice-0 que a gente já
       tinha consertado pro empate saturado, escancarado no caso "tudo zero". Medido: o
       cérebro-zero CONQUISTOU o HyperNEAT (31 de 39 provados, mediana 0 conexões) — não por
       ser estratégia, mas por bug de desempate. Quem não paga por um cérebro não age.
    2) EMPATE SATURADO (topo >=0.9 e várias coladas nele) -> sorteio uniforme. O cérebro grita
       tudo ao mesmo tempo e genuinamente não distingue; escolher por índice seria viés.
    3) Decisão graduada ou vencedor claro -> argmax, respeitando o gradiente.
    """
    mx = max(out)
    if max(abs(mx), abs(min(out))) < NULL_EPS:
        return 4                                    # "stay": o cérebro não disse nada
    near = [i for i in range(len(out)) if out[i] >= mx - 0.05]
    if len(near) > 1 and mx >= 0.9:
        return random.choice(near)
    return max(range(len(out)), key=lambda i: out[i])


async def run_one(idx: int):
    while True:
        try:
            # CRONÔMETRO DE CICLO (diagnóstico Fable 04/08): 50 clientes sustentavam só
            # ~8 amebas — cada cliente passava 79% do ciclo FORA DO AR (~36s de 46s).
            # Hipótese principal: o close handshake do websocket (close_timeout padrão
            # 10s na lib) esperando o frame de um servidor que já saiu do handler na
            # morte. close_timeout=1 mitiga; o print mede connect/vida/close p/ provar.
            t0 = time.perf_counter()
            t_born = t_dead = None
            async with websockets.connect(URL, max_size=8_000_000, ssl=SSL,
                                          close_timeout=1) as ws:
                welcome = json.loads(await ws.recv())
                t_born = time.perf_counter()
                seed_a = welcome.get("brain_a")
                seed_b = welcome.get("brain_b")
                body = welcome.get("body") or welcome.get("stats") or {}
                stomach_size = body.get("stomach_size", 200) or 200

                # HERANÇA: 2 pais -> cruzamento sexual + mutação; 1 -> só mutação (bootstrap
                # assexuado enquanto o banco não tem 2 provados); 0 -> primordial. O crossover é
                # o que MISTURA linhagens de clientes/máquinas diferentes = diversidade no mundo
                # distribuído. (BUG estrutural conhecido: inovação/id de nó são numerados por
                # processo, então linhagens não alinham e o crossover erode genes + gera warnings.
                # Fix correto = numeração GLOBAL/determinística de id de nó; NÃO remover o sexo.)
                if seed_a and seed_b:
                    g = nb.crossover(nb.unpack(seed_a), nb.unpack(seed_b), random.randint(1, 1_000_000))
                    nb.mutate(g)
                    origin = "cruzamento"
                elif seed_a or seed_b:
                    g = nb.unpack(seed_a or seed_b)
                    nb.mutate(g)
                    origin = "mutacao"
                else:
                    g = nb.random_genome(random.randint(1, 1_000_000))
                    origin = "primordial"

                # reporta o GENOMA final (compactado) + complexidade (telemetria pro mundo logar,
                # sem ele precisar decodificar o blob — respeita "cérebro opaco"). O mundo envolve
                # com genealogia+assinatura e guarda.
                # nodes/conns = genoma (total / habilitadas): custo §15.3. genes = §21.
                # fnodes/fconns = CÉREBRO REAL (funcional): observabilidade + ACUIDADE (§24 #5):
                # antes a acuidade era alimentada pelas habilitadas — 97,5% tecido morto ligando
                # a visão de graça. Agora só o que computa enxerga. Num genoma sadio fconns≈conns
                # (nascer magro é 100% funcional), então a escala não muda — só para de mentir.
                nodes, conns = nb.complexity(g)
                fnodes, fconns = nb.functional_complexity(g)
                genes = len(g.nodes) + len(g.connections)
                acuity = acuity_params(fconns)   # (PSF, sigma, A) — fixo em vida
                await ws.send(json.dumps({"type": "brain", "brain": nb.pack(g),
                                          "nodes": nodes, "conns": conns,
                                          "fnodes": fnodes, "fconns": fconns, "genes": genes,
                                          "acuity": round(acuity[2], 3)}))
                net = nb.build_net(g)
                _telemetry(idx, origin, nodes, conns, fnodes, fconns, genes, acuity[2])
                print(f"[{idx}] nasceu ({origin}) nos={nodes} lig={conns} "
                      f"real={fnodes}/{fconns} genes={genes} "
                      f"acuidade={acuity[2]:.2f} sigma={acuity[1]:.2f}")

                viz_sent = False   # já mandei a ESTRUTURA nesta sessão de observação?
                out_keys = set(nb.load_config().genome_config.output_keys)  # ids dos nós de saída
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "UPDATE":
                        if not msg.get("alive", True):
                            t_dead = time.perf_counter()
                            break  # morreu -> reconecta
                        continue
                    if "vision" in msg:  # TICK: decide e age
                        # R10: corpo atual (tanque cresce com a massa). Sem o campo, mantém WELCOME.
                        stomach_size = msg.get("stomach_size", stomach_size)
                        energy = msg.get("energy", 0)
                        stomach = msg.get("stomach", 0)
                        # R4/#3: ingested vem do TICK (fato do mundo; 0.0 na ausência do campo).
                        # Sem estado, sem decaimento: um tick não vaza para o seguinte.
                        inp = encode(msg.get("vision"), energy, stomach, stomach_size,
                                     msg.get("ingested", 0.0),
                                     msg.get("pace_sin", 0.0), msg.get("pace_cos", 0.0), acuity,
                                     damage=msg.get("damage", 0.0), impact=msg.get("impact", 0.0),
                                     moved_self=msg.get("moved_self", 0.0),
                                     moved_passive=msg.get("moved_passive", 0.0),
                                     contact_body=msg.get("contact_body", 0.0),
                                     contact_wall=msg.get("contact_wall", 0.0))
                        out = net.activate(inp)
                        a = decide(out)
                        await ws.send(json.dumps(ACTIONS[a]))

                        # VIZ DE CÉREBRO: se algum viewer observa esta ameba, manda estrutura (1x) +
                        # ativações (todo tick, 4 Hz). net.values tem os valores de TODOS os nós após
                        # o activate — de graça. O mundo só relaya. Sem observador, não custa nada.
                        if msg.get("viz"):
                            act = {
                                "inp": [round(x, 3) for x in inp],                       # 192 entradas (já borradas)
                                "hid": {str(n): round(net.values.get(n, 0.0), 3)         # ocultos
                                        for n in g.nodes if n not in out_keys},
                                "out": [round(x, 3) for x in out],                       # 7 saídas
                                "win": a,                                                 # ação vencedora
                            }
                            payload = {"type": "brain_viz", "act": act}
                            if not viz_sent:
                                payload["struct"] = nb.to_dict(g)   # topologia + pesos, uma vez
                                viz_sent = True
                            await ws.send(json.dumps(payload))
                        else:
                            viz_sent = False   # parou de observar -> reenvia estrutura na próxima
            t_end = time.perf_counter()
            print(f"[{idx}] ciclo: connect {((t_born or t0) - t0):.1f}s | "
                  f"vida {((t_dead - t_born) if (t_dead and t_born) else -1):.1f}s | "
                  f"close {(t_end - (t_dead or t_born or t0)):.1f}s")
        except Exception as e:
            print(f"[{idx}] reconnect ({e.__class__.__name__}: {e})")
            await asyncio.sleep(1.0)


async def main():
    print(f"Executor nativo: {N} amebas -> {URL}")
    await asyncio.gather(*[run_one(i) for i in range(N)])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecutor nativo encerrado.")

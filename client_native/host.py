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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000"
OP = os.getenv("REGENES_OPERATOR", "")  # dono da linhagem (carimbo na genealogia)
URL = (BASE.rstrip("/") + "/ws/join?species=Native_NEAT&paradigm=neuroevolution_topology"
       "&wants_brain=1&self_learns=0" + (f"&operator={OP}" if OP else ""))

# TELEMETRIA LOCAL de complexidade do cérebro (a produção só guarda sumários; isto dá a curva
# na hora, sem depender de deploy do mundo). 1 linha por nascimento. Append síncrono é seguro no
# asyncio single-thread (sem await no meio). Desligar com REGENES_TELEMETRY=0.
_TELEMETRY = os.path.join(os.path.dirname(__file__), "native_telemetry.csv")
_TELEMETRY_ON = os.getenv("REGENES_TELEMETRY", "1") != "0"


def _telemetry(idx: int, origin: str, nodes: int, conns: int, acuity: float) -> None:
    if not _TELEMETRY_ON:
        return
    try:
        new = not os.path.exists(_TELEMETRY)
        with open(_TELEMETRY, "a", encoding="ascii") as f:
            if new:
                f.write("unix_time,idx,origin,nodes,conns,acuity\n")
            f.write(f"{time.time():.0f},{idx},{origin},{nodes},{conns},{acuity:.3f}\n")
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
ACUITY_PRED_LO = 0.40   # predação (inimigo/perigo) entra em FADE contínuo de A=0.40...
ACUITY_PRED_HI = 0.70   # ...até 100% em A=0.70 (sem degrau)
_PRED_CH = (2, 3)       # canais de predação: 2=inimigo, 3=perigo. 0=obstáculo,1=cheiro,4=comida = sempre.


def acuity_params(conns):
    """C (conexões) -> (kernel_gaussiano, raio, peso_predacao, A). Fixo no nascimento."""
    A = conns / (conns + ACUITY_K)
    sigma = ACUITY_SIGMA_MAX * (1.0 - A)
    kernel, r = _blur_kernel(sigma)
    pred_w = max(0.0, min(1.0, (A - ACUITY_PRED_LO) / (ACUITY_PRED_HI - ACUITY_PRED_LO)))
    return (kernel, r, pred_w, A)


def _blur_kernel(sigma):
    """Kernel gaussiano 1D normalizado (raio ~3σ). σ quase 0 -> ([1.0], 0) = visão nítida."""
    if sigma < 0.35:
        return ([1.0], 0)
    r = max(1, int(round(3.0 * sigma)))
    ker = [math.exp(-(d * d) / (2.0 * sigma * sigma)) for d in range(-r, r + 1)]
    s = sum(ker)
    return ([w / s for w in ker], r)


def _blur(row, kernel, r):
    """Convolve o cone de 31 células com o kernel (borda: clamp). r=0 -> identidade (nítido)."""
    if r == 0:
        return list(row[:31])
    out = [0.0] * 31
    for k in range(31):
        acc = 0.0
        for j in range(len(kernel)):
            idx = k + j - r
            idx = 0 if idx < 0 else (30 if idx > 30 else idx)
            acc += kernel[j] * row[idx]
        out[k] = acc
    return out


# Encoding v3 EGOCÊNTRICO: 161 = bias + energy + stomach + endorfina + MARCA-PASSO(sin,cos)
# + 5 canais x 31 do CONE, BORRADOS pela acuidade do cérebro (desfoque contínuo; predação em fade).
def encode(vision, energy, stomach, stomach_size, endo, pace_sin, pace_cos, acuity):
    if not vision or len(vision) < 5 or len(vision[0]) < 31:
        return [0.0] * 161
    kernel, r, pred_w = acuity[0], acuity[1], acuity[2]
    ss = stomach_size or 1.0
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, endo / 100.0, pace_sin, pace_cos]
    for ch in range(5):
        blurred = _blur(vision[ch], kernel, r)
        if ch in _PRED_CH and pred_w < 1.0:
            blurred = [v * pred_w for v in blurred]      # predação em fade-in contínuo
        inp.extend(blurred)
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
            async with websockets.connect(URL, max_size=8_000_000, ssl=SSL) as ws:
                welcome = json.loads(await ws.recv())
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
                nodes, conns = nb.complexity(g)
                acuity = acuity_params(conns)   # (L, B, ve_predacao, A) — fixo em vida
                await ws.send(json.dumps({"type": "brain", "brain": nb.pack(g),
                                          "nodes": nodes, "conns": conns, "acuity": round(acuity[3], 3)}))
                net = nb.build_net(g)
                _telemetry(idx, origin, nodes, conns, acuity[3])
                print(f"[{idx}] nasceu ({origin}) nos={nodes} lig={conns} "
                      f"acuidade={acuity[3]:.2f} (niveis={acuity[0]} bins={acuity[1]} predacao={acuity[2]})")

                endo, last_e = 50.0, None
                viz_sent = False   # já mandei a ESTRUTURA nesta sessão de observação?
                out_keys = set(nb.load_config().genome_config.output_keys)  # ids dos nós de saída
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "UPDATE":
                        if not msg.get("alive", True):
                            break  # morreu -> reconecta
                        e = msg.get("energy")
                        if e is not None:
                            if last_e is not None and e > last_e:
                                endo = min(100.0, endo + 100.0)  # comeu -> pico de endorfina
                            last_e = e
                        continue
                    if "vision" in msg:  # TICK: decide e age
                        energy = msg.get("energy", 0)
                        stomach = msg.get("stomach", 0)
                        endo -= 0.2
                        if energy < stomach_size * 0.5:
                            endo -= 2.0
                        endo = max(0.0, min(100.0, endo))
                        inp = encode(msg.get("vision"), energy, stomach, stomach_size, endo,
                                     msg.get("pace_sin", 0.0), msg.get("pace_cos", 0.0), acuity)
                        out = net.activate(inp)
                        a = decide(out)
                        await ws.send(json.dumps(ACTIONS[a]))

                        # VIZ DE CÉREBRO: se algum viewer observa esta ameba, manda estrutura (1x) +
                        # ativações (todo tick, 4 Hz). net.values tem os valores de TODOS os nós após
                        # o activate — de graça. O mundo só relaya. Sem observador, não custa nada.
                        if msg.get("viz"):
                            act = {
                                "inp": [round(x, 3) for x in inp],                       # 161 entradas (já borradas)
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

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
# "Você enxerga na resolução que seu cérebro consegue processar." Cérebro pobre vê grosso
# (binário, cone em blocões, sem predação); afina sozinho conforme a linhagem evolui complexidade.
# É lei de embodiment (como cubo-quadrado/Kleiber), não currículo agendado. Calculada no cliente,
# no nascimento (cérebro é fixo em vida), a partir do C do próprio genoma — sem reset, 161 intacto.
ACUITY_K = 400.0     # meia-saturação: A = C/(C+K)
ACUITY_LMAX = 16     # teto de níveis de intensidade (binário -> contínuo)
ACUITY_BMIN = 3      # grão espacial mínimo do cone (de 31 células)
ACUITY_APRED = 0.5   # acuidade que destrava os canais de predação
_PRED_CH = (2, 3)    # canais gated: 2=inimigo, 3=perigo (predação). 0=obstáculo,1=cheiro,4=comida = sempre.


def acuity_params(conns):
    """C (conexões) -> (n_niveis_intensidade, n_bins_espaciais, ve_predacao, A). Fixo no nascimento."""
    A = conns / (conns + ACUITY_K)
    L = max(2, round(2 + A * A * (ACUITY_LMAX - 2)))            # A^2 segura no binário mais tempo
    B = max(ACUITY_BMIN, min(31, round(ACUITY_BMIN + A * A * (31 - ACUITY_BMIN))))
    return (L, B, A > ACUITY_APRED, A)


def _pool31(row, B):
    """31 células -> B blocos (média) -> reamostra de volta pros 31 slots (mantém num_inputs)."""
    if B >= 31:
        return list(row[:31])
    out = [0.0] * 31
    for k in range(31):
        b = (k * B) // 31                    # bloco a que a célula k pertence
        lo, hi = (b * 31) // B, ((b + 1) * 31) // B
        seg = row[lo:hi] or [row[k]]
        out[k] = sum(seg) / len(seg)
    return out


def _quantize(v, L):
    """Valor em [0,1] -> um de L níveis (L=2 => binário; L grande => ~contínuo)."""
    if L >= 256:
        return v
    return round(v * (L - 1)) / (L - 1)


# Encoding do executor nativo (v3 EGOCÊNTRICO): 161 = bias + energy + stomach + endorfina
# + MARCA-PASSO(sin,cos) + 5 canais x 31 células do CONE, filtrados pela ACUIDADE do cérebro.
def encode(vision, energy, stomach, stomach_size, endo, pace_sin, pace_cos, acuity):
    if not vision or len(vision) < 5 or len(vision[0]) < 31:
        return [0.0] * 161
    L, B, see_pred = acuity[0], acuity[1], acuity[2]
    ss = stomach_size or 1.0
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, endo / 100.0, pace_sin, pace_cos]
    for ch in range(5):
        if ch in _PRED_CH and not see_pred:
            inp.extend([0.0] * 31)                       # canal de predação ainda escuro (invisível)
            continue
        pooled = _pool31(vision[ch], B)                  # resolução espacial (cone em blocões)
        inp.extend(_quantize(v, L) for v in pooled)      # resolução de intensidade (binário->contínuo)
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
                        out = net.activate(encode(msg.get("vision"), energy, stomach, stomach_size, endo,
                                                  msg.get("pace_sin", 0.0), msg.get("pace_cos", 0.0), acuity))
                        a = max(range(len(out)), key=lambda i: out[i])
                        await ws.send(json.dumps(ACTIONS[a]))
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

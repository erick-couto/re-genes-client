"""
host_hyper.py — Executor da espécie HyperNEAT (segunda espécie da arena).

Mesmo contrato do client_native (cérebro-no-genoma, o mundo seleciona), mas com ENCODING
INDIRETO: o genoma não é a rede — é um CPPN que PINTA a rede a partir da geometria.

  1. recebe do mundo a semente (CPPN do pai) — só de HyperNEAT (o mundo garante o
     isolamento reprodutivo, §15: espécies não cruzam);
  2. cruza + muta o CPPN (reusa a maquinaria de genoma do client_native, inclusive a
     identidade determinística — o CPPN é, ele mesmo, um genoma NEAT);
  3. EXPRESSA: consulta o CPPN em cada par de coordenadas -> substrato 161->16->7;
  4. roda o forward pass do substrato a cada tick;
  5. ao morrer, reconecta -> nova ameba.

A aposta: o cone TEM geometria, e o NEAT direto é cego a ela — precisa sortear ~31 fios
independentes pra formar "comida à esquerda -> vira esquerda". O CPPN expressa isso como UMA
função de x1*x2. Se a aposta estiver certa, esta espécie acha quimiotaxia muito mais rápido.
Quem decide não sou eu: as duas competem no mesmo mundo. Ver GENESIS_BIBLE §15.

Uso:
    python host_hyper.py [N] [ws_base]
      N       -> quantas amebas HyperNEAT (default 8)
      ws_base -> ex.: ws://127.0.0.1:8000 (default)
"""
import asyncio
import json
import math
import os
import random
import ssl
import sys

import websockets

# reusa a maquinaria de genoma do client nativo (o CPPN É um genoma NEAT: pack/unpack/
# crossover/mutate valem igual — inclusive a identidade DETERMINÍSTICA, sem a qual o
# crossover distribuído erode, que foi o bug estrutural que a gente caçou lá atrás).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "client_native"))
import neat_brain as nb          # noqa: E402
import substrate as sub          # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000"
OP = os.getenv("REGENES_OPERATOR", "")
URL = (BASE.rstrip("/") + "/ws/join?species=HyperNEAT&paradigm=hyperneat_cppn"
       "&wants_brain=1&self_learns=0" + (f"&operator={OP}" if OP else ""))
SSL = ssl.create_default_context() if BASE.startswith("wss") else None

_CPPN_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config-cppn")

# --- lei da acuidade: idêntica à do nativo (mesma física de percepção, §FISICA_DA_PERCEPCAO).
# A capacidade C aqui é a do SUBSTRATO EXPRESSO (a rede que de fato pensa), não a do CPPN:
# a lei fala de capacidade neural, e o substrato é o sistema nervoso. Efeito colateral honesto
# e importante: o encoding indireto expressa MUITA conexão barato -> nasce enxergando melhor
# que o NEAT direto. Isso é a força do paradigma, não trapaça — mas confunde a comparação e
# está registrado como ressalva na §15.
ACUITY_K = 120.0
ACUITY_SIGMA_MAX = 6.0
ACUITY_PRED_LO, ACUITY_PRED_HI = 0.40, 0.70
_PRED_CH = (2, 3)


def acuity_params(conns):
    A = conns / (conns + ACUITY_K)
    sigma = ACUITY_SIGMA_MAX * (1.0 - A)
    kernel, r = _blur_kernel(sigma)
    pred_w = max(0.0, min(1.0, (A - ACUITY_PRED_LO) / (ACUITY_PRED_HI - ACUITY_PRED_LO)))
    return (kernel, r, pred_w, A)


def _blur_kernel(sigma):
    if sigma < 0.35:
        return ([1.0], 0)
    r = max(1, int(round(3.0 * sigma)))
    ker = [math.exp(-(d * d) / (2.0 * sigma * sigma)) for d in range(-r, r + 1)]
    s = sum(ker)
    return ([w / s for w in ker], r)


def _blur(row, kernel, r):
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


def encode(vision, energy, stomach, stomach_size, endo, pace_sin, pace_cos, acuity):
    """IDÊNTICO ao do nativo — mesma ordem das 161 entradas. Tem que ser: o substrato mapeia
    coordenada por ÍNDICE (substrate.INPUT_COORDS segue esta mesma ordem)."""
    if not vision or len(vision) < 5 or len(vision[0]) < 31:
        return [0.0] * 161
    kernel, r, pred_w = acuity[0], acuity[1], acuity[2]
    ss = stomach_size or 1.0
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, endo / 100.0, pace_sin, pace_cos]
    for ch in range(5):
        blurred = _blur(vision[ch], kernel, r)
        if ch in _PRED_CH and pred_w < 1.0:
            blurred = [v * pred_w for v in blurred]
        inp.extend(blurred)
    return inp


ACTIONS = [
    {"action": "forward"},
    {"action": "backward"},
    {"action": "turn", "dir": "left"},
    {"action": "turn", "dir": "right"},
    {"action": "stay"},
    {"action": "attack"},
    {"action": "push"},
]


async def run_one(idx: int):
    while True:
        try:
            async with websockets.connect(URL, max_size=8_000_000, ssl=SSL) as ws:
                welcome = json.loads(await ws.recv())
                seed_a, seed_b = welcome.get("brain_a"), welcome.get("brain_b")
                body = welcome.get("body") or welcome.get("stats") or {}
                stomach_size = body.get("stomach_size", 200) or 200

                # HERANÇA do CPPN. O mundo já filtra por espécie (§15), então estas sementes
                # são sempre CPPNs — nunca um genoma do NEAT direto (que nem decodificaria).
                if seed_a and seed_b:
                    g = nb.crossover(nb.unpack(seed_a), nb.unpack(seed_b),
                                     random.randint(1, 1_000_000))
                    nb.mutate(g)
                    origin = "cruzamento"
                elif seed_a or seed_b:
                    g = nb.unpack(seed_a or seed_b)
                    nb.mutate(g)
                    origin = "mutacao"
                else:
                    g = nb.random_genome(random.randint(1, 1_000_000))
                    origin = "primordial"

                # EXPRESSÃO: o CPPN pinta o substrato. Custo pago 1x, no nascimento.
                cppn = nb.build_net(g)
                W_ih, W_ho, n_conns = sub.express(cppn)
                cppn_nodes, cppn_conns = nb.complexity(g)
                acuity = acuity_params(n_conns)

                # Reporta o CPPN (o genoma) como blob opaco. nodes/conns = do SUBSTRATO (a rede
                # que pensa), pra a telemetria do mundo comparar maçã com maçã com o nativo.
                await ws.send(json.dumps({
                    "type": "brain", "brain": nb.pack(g),
                    "nodes": sub.N_IN + sub.N_HID + sub.N_OUT, "conns": n_conns,
                    "acuity": round(acuity[3], 3)}))
                print(f"[H{idx}] nasceu ({origin}) cppn: {cppn_nodes}n/{cppn_conns}c -> "
                      f"substrato: {n_conns} sinapses | acuidade={acuity[3]:.2f}")

                endo, last_e = 50.0, None
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "UPDATE":
                        if not msg.get("alive", True):
                            break
                        e = msg.get("energy")
                        if e is not None:
                            if last_e is not None and e > last_e:
                                endo = min(100.0, endo + 100.0)   # comeu -> endorfina
                            last_e = e
                        continue
                    if "vision" in msg:
                        energy = msg.get("energy", 0)
                        stomach = msg.get("stomach", 0)
                        endo -= 0.2
                        if energy < stomach_size * 0.5:
                            endo -= 2.0
                        endo = max(0.0, min(100.0, endo))
                        inp = encode(msg.get("vision"), energy, stomach, stomach_size, endo,
                                     msg.get("pace_sin", 0.0), msg.get("pace_cos", 0.0), acuity)
                        out = sub.activate(W_ih, W_ho, inp)
                        # desempate por saturação: igual ao nativo (§14.5) — no empate saturado
                        # o cérebro não distingue, então sorteia; senão respeita o gradiente.
                        mx = max(out)
                        near = [i for i in range(len(out)) if out[i] >= mx - 0.05]
                        a = random.choice(near) if (len(near) > 1 and mx >= 0.9) \
                            else max(range(len(out)), key=lambda i: out[i])
                        await ws.send(json.dumps(ACTIONS[a]))
        except Exception as e:
            print(f"[H{idx}] reconnect ({e.__class__.__name__}: {e})")
            await asyncio.sleep(1.0)


async def main():
    nb.load_config(_CPPN_CONFIG)   # memoiza O CONFIG DO CPPN neste processo (7 in / 2 out)
    print(f"Executor HyperNEAT: {N} amebas -> {URL}")
    print(f"substrato: {sub.N_IN} entradas -> {sub.N_HID} ocultos -> {sub.N_OUT} saidas "
          f"| {sub.N_IN*sub.N_HID + sub.N_HID*sub.N_OUT} sinapses possiveis")
    await asyncio.gather(*[run_one(i) for i in range(N)])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecutor HyperNEAT encerrado.")

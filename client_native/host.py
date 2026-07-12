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

import websockets
import neat_brain as nb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000"
OP = os.getenv("REGENES_OPERATOR", "")  # dono da linhagem (carimbo na genealogia)
URL = (BASE.rstrip("/") + "/ws/join?species=Native_NEAT&paradigm=neuroevolution_topology"
       "&wants_brain=1&self_learns=0" + (f"&operator={OP}" if OP else ""))


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

# Mesmo encoding do client NEAT (arena justa): 104 = bias + energy + stomach + endorfina
# + 4 canais x recorte 5x5 central. Canal 0 (obstáculos) é binarizado.
def encode(vision, energy, stomach, stomach_size, endo):
    if not vision or len(vision) < 4 or len(vision[0]) < 9:
        return [0.0] * 104
    ss = stomach_size or 1.0
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, endo / 100.0]
    C = 4  # centro do 9x9
    for ch in range(4):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                v = vision[ch][C + dy][C + dx]
                inp.append(1.0 if (ch == 0 and v > 0) else v)
    return inp

# índice -> comando de wire (bate com ACTION_SPEC do mundo)
ACTIONS = [
    {"action": "move", "direction": "UP"},
    {"action": "move", "direction": "DOWN"},
    {"action": "move", "direction": "LEFT"},
    {"action": "move", "direction": "RIGHT"},
    {"action": "stay"},
    {"action": "attack", "direction": "UP"},
    {"action": "attack", "direction": "DOWN"},
    {"action": "attack", "direction": "LEFT"},
    {"action": "attack", "direction": "RIGHT"},
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

                # HERANÇA: 2 pais -> cruzamento sexual + mutação; 1 -> só mutação; 0 -> primordial.
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

                # reporta o GENOMA final (compactado); o mundo envolve com genealogia+assinatura
                await ws.send(json.dumps({"type": "brain", "brain": nb.pack(g)}))
                net = nb.build_net(g)
                nodes, conns = nb.complexity(g)
                print(f"[{idx}] nasceu ({origin}) cerebro nos={nodes} lig={conns}")

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
                        out = net.activate(encode(msg.get("vision"), energy, stomach, stomach_size, endo))
                        a = max(range(len(out)), key=lambda i: out[i])
                        await ws.send(json.dumps(ACTIONS[a]))
        except Exception as e:
            print(f"[{idx}] reconnect ({e.__class__.__name__}: {e})")
            await asyncio.sleep(1.0)


async def main():
    print(f"Executor nativo: {N} amebas -> {URL}")
    await asyncio.gather(*[run_one(i) for i in range(N)])


if __name__ == "__main__":
    asyncio.run(main())

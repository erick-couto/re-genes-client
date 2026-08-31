#!/usr/bin/env python3
"""Régua de quimiotaxia — CONTROLE, não participante.

Uma (ou N) amebas que seguem o cheiro de contato (§52) e pastam o que estiver
embaixo. Não aprende, não evolui, wants_brain=0, self_learns=0. O cérebro NÃO
entra no brain_bank. NÃO é anunciada como competidora da arena.

Uso:
    REGENES_SERVER=ws://127.0.0.1:8081 python client_regua_scent.py [N]

N default = 1. Não coloque isto no start_luna.sh dos executores Fase 2.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
import sys
import time

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regua_scent import DietFilter, decide, FWD, BACK, TURN_L, TURN_R, STAY, ATTACK, PUSH

SPECIES = "ReguaScent"
PARADIGM = "control_heuristic"
PROTOCOL_VERSION = 8
N_OBS = 163
N_ACTIONS = 8

BASE = os.getenv("REGENES_SERVER", sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8081")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OP = os.getenv("REGENES_OPERATOR", "regua")

URL = (BASE.rstrip("/")
       + f"/ws/join?species={SPECIES}&paradigm={PARADIGM}"
       + "&wants_brain=0&self_learns=0"
       + f"&protocol_version={PROTOCOL_VERSION}&n_obs={N_OBS}&n_actions={N_ACTIONS}"
       + f"&operator={OP}")

ACTIONS = [
    {"action": "forward"},
    {"action": "backward"},
    {"action": "turn", "dir": "left"},
    {"action": "turn", "dir": "right"},
    {"action": "stay"},
    {"action": "attack"},
    {"action": "push"},
    {"action": "bite"},
]
assert len(ACTIONS) == N_ACTIONS
# A régua decide entre as 7 ações originais — instrumento de medição não emite bocado
# (declarar 8 no join sem nunca emitir é honesto: o shape é do protocolo, não da dieta dela).
assert (FWD, BACK, TURN_L, TURN_R, STAY, ATTACK, PUSH) == tuple(range(7))


def _ssl():
    if not URL.startswith("wss"):
        return None
    ctx = ssl.create_default_context()
    if os.getenv("REGENES_INSECURE_TLS") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


SSL = _ssl()


async def run_one(idx: int):
    while True:
        diet = DietFilter()
        rng = random.Random()
        ate = 0
        ticks = 0
        t0 = time.perf_counter()
        try:
            async with websockets.connect(URL, max_size=8_000_000, ssl=SSL,
                                          close_timeout=1) as ws:
                welcome = json.loads(await ws.recv())
                my_id = welcome.get("id")
                print(f"[{idx}] nasceu {my_id}  CONTROLE {SPECIES}  "
                      f"wants_brain=0  não-participante", flush=True)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "UPDATE":
                        if not msg.get("alive", True):
                            dt = time.perf_counter() - t0
                            print(f"[{idx}] morreu {my_id} ticks={ticks} "
                                  f"ingestoes_vistas={ate} vida={dt:.1f}s",
                                  flush=True)
                            break
                        continue
                    if msg.get("type") != "TICK" and "vision" not in msg:
                        continue
                    food0 = 0.0
                    vis = msg.get("vision") or []
                    if vis and len(vis) > 3 and vis[3]:
                        food0 = float(vis[3][0])
                    ingested = float(msg.get("ingested") or 0.0)
                    diet.update(food0, ingested,
                                float(msg.get("stomach") or 0.0),
                                float(msg.get("stomach_size") or 0.0))
                    if ingested > 0:
                        ate += 1
                    a = decide(vis, msg.get("chemical"), diet, rng)
                    await ws.send(json.dumps(ACTIONS[a]))
                    ticks += 1
        except Exception as e:
            print(f"[{idx}] reconnect ({e.__class__.__name__}: {e})", flush=True)
        await asyncio.sleep(1.0)


async def main():
    print(f"RÉGUA (controle, não participante): {N}x {SPECIES} -> {URL}",
          flush=True)
    await asyncio.gather(*[run_one(i) for i in range(N)])


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nrégua encerrada.")

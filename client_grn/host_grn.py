"""
host_grn.py — Executor da espécie GRN (terceira espécie da arena).

Mesmo contrato WS dos outros (v7 / 163 / 7, wants_brain=1, self_learns=0).
O genoma é uma sopa regulatória (`grn_brain.py`): concentrações que vazam entre
ticks, ligações que podem ciclar. Sem fitness. O mundo isola o acasalamento
por `species=GRN` (§15).

Uso:
    python host_grn.py [N] [ws_base]
"""
import asyncio
import json
import os
import random
import ssl
import sys
import time

import websockets

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "client_native"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cone_psf  # noqa: E402
import grn_brain as gb  # noqa: E402
from decide_action import decide  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000"
OP = os.getenv("REGENES_OPERATOR", "")
PROTOCOL_VERSION = 7
N_OBS = 163
N_ACTIONS = 7
URL = (BASE.rstrip("/") + "/ws/join?species=GRN&paradigm=gene_regulatory_network"
       "&wants_brain=1&self_learns=0"
       f"&protocol_version={PROTOCOL_VERSION}&n_obs={N_OBS}&n_actions={N_ACTIONS}"
       + (f"&operator={OP}" if OP else ""))

_TELEMETRY = os.path.join(os.path.dirname(__file__), "grn_telemetry.csv")
_TELEMETRY_ON = os.getenv("REGENES_TELEMETRY", "1") != "0"

ACUITY_K = 120.0
ACUITY_SIGMA_MAX = 6.0


def _telemetry(idx, origin, nodes, conns, fnodes, fconns, genes, acuity):
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
        pass


def _ssl_ctx():
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


def acuity_params(conns):
    """Mesma lei de percepção dos outros: A = C/(C+K). C = fconns da sopa."""
    A = conns / (conns + ACUITY_K)
    sigma = ACUITY_SIGMA_MAX * (1.0 - A)
    return (cone_psf.psf(sigma), sigma, A)


def _blur(row, P):
    return cone_psf.blur(row, P)


def encode(vision, chemical, energy, stomach, stomach_size, ingested, pace_sin, pace_cos,
           acuity,
           damage=0.0, impact=0.0,
           moved_self=0.0, moved_passive=0.0, contact_body=0.0, contact_wall=0.0):
    """IDÊNTICO ao Native/Hyper — o mundo fala uma língua só."""
    if not vision or len(vision) < 4 or len(vision[0]) < 31:
        return [0.0] * 163
    if not chemical or len(chemical) < 3 or len(chemical[0]) < 9:
        return [0.0] * 163
    P = acuity[0]
    ss = stomach_size or 1.0
    inp = [1.0, min(1.0, energy / ss), min(stomach, ss) / ss, min(1.0, ingested / ss),
           pace_sin, pace_cos, min(1.0, damage / ss), min(1.0, impact / ss),
           moved_self, moved_passive, contact_body, contact_wall]
    for ch in range(4):
        inp.extend(_blur(vision[ch], P))
    for ch in range(3):
        inp.extend(chemical[ch])
    return inp


def _viz_src(i):
    """IDs da sopa → IDs que o painel Native já lê (input negativo, saída 0..6)."""
    if i < gb.N_IN:
        return -(i + 1)
    if gb.OUT0 <= i < gb.OUT0 + gb.N_OUT:
        return i - gb.OUT0
    return i


def _viz_struct(g):
    return {
        "nh": g.nh,
        "conns": [[_viz_src(s), _viz_src(d), w, int(en)] for s, d, w, en in g.regs],
    }


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
            t0 = time.perf_counter()
            t_born = t_dead = None
            async with websockets.connect(URL, max_size=8_000_000, ssl=SSL,
                                          close_timeout=1) as ws:
                welcome = json.loads(await ws.recv())
                t_born = time.perf_counter()
                seed_a, seed_b = welcome.get("brain_a"), welcome.get("brain_b")
                body = welcome.get("body") or welcome.get("stats") or {}
                stomach_size = body.get("stomach_size", 200) or 200

                if seed_a and seed_b:
                    g = gb.crossover(gb.unpack(seed_a), gb.unpack(seed_b),
                                     random.randint(1, 1_000_000))
                    g = gb.mutate(g)
                    origin = "cruzamento"
                elif seed_a or seed_b:
                    g = gb.mutate(gb.unpack(seed_a or seed_b))
                    origin = "mutacao"
                else:
                    g = gb.random_genome(random.randint(1, 1_000_000))
                    origin = "primordial"

                nodes, conns = gb.complexity(g)
                fnodes, fconns = gb.functional_complexity(g)
                genes = gb.genes_count(g)
                acuity = acuity_params(fconns)
                await ws.send(json.dumps({
                    "type": "brain", "brain": gb.pack(g),
                    "nodes": nodes, "conns": conns,
                    "fnodes": fnodes, "fconns": fconns, "genes": genes,
                    "acuity": round(acuity[2], 3),
                }))
                soup = gb.Soup(g)
                _telemetry(idx, origin, nodes, conns, fnodes, fconns, genes, acuity[2])
                print(f"[G{idx}] nasceu ({origin}) genes_reg={nodes} lig={conns} "
                      f"real={fnodes}/{fconns} genes={genes} "
                      f"acuidade={acuity[2]:.2f} sigma={acuity[1]:.2f}")

                viz_sent = False
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "UPDATE":
                        if not msg.get("alive", True):
                            t_dead = time.perf_counter()
                            break
                        continue
                    if "vision" in msg:
                        stomach_size = msg.get("stomach_size", stomach_size)
                        inp = encode(msg.get("vision"), msg.get("chemical"),
                                     msg.get("energy", 0), msg.get("stomach", 0),
                                     stomach_size,
                                     msg.get("ingested", 0.0),
                                     msg.get("pace_sin", 0.0), msg.get("pace_cos", 0.0),
                                     acuity,
                                     damage=msg.get("damage", 0.0),
                                     impact=msg.get("impact", 0.0),
                                     moved_self=msg.get("moved_self", 0.0),
                                     moved_passive=msg.get("moved_passive", 0.0),
                                     contact_body=msg.get("contact_body", 0.0),
                                     contact_wall=msg.get("contact_wall", 0.0))
                        out = soup.step(inp)
                        a = decide(out)
                        await ws.send(json.dumps(ACTIONS[a]))
                        if msg.get("viz"):
                            act = {
                                "inp": [round(x, 3) for x in inp],
                                "hid": {str(k): round(v, 3) for k, v in soup.conc.items()
                                        if k >= gb.HID0},
                                "out": [round(x, 3) for x in out],
                                "win": a,
                            }
                            payload = {"type": "brain_viz", "act": act}
                            if not viz_sent:
                                payload["struct"] = _viz_struct(g)
                                viz_sent = True
                            await ws.send(json.dumps(payload))
                        else:
                            viz_sent = False
            t_end = time.perf_counter()
            print(f"[G{idx}] ciclo: connect {((t_born or t0) - t0):.1f}s | "
                  f"vida {((t_dead - t_born) if (t_dead and t_born) else -1):.1f}s | "
                  f"close {(t_end - (t_dead or t_born or t0)):.1f}s")
        except Exception as e:
            print(f"[G{idx}] reconnect ({e.__class__.__name__}: {e})")
            await asyncio.sleep(1.0)


async def main():
    print(f"Executor GRN: {N} amebas -> {URL}")
    await asyncio.gather(*[run_one(i) for i in range(N)])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecutor GRN encerrado.")

# -*- coding: utf-8 -*-
"""Sopa regulatória — o cérebro GRN (terceira espécie).

Não é NEAT: não há camadas, o grafo pode ter ciclos, o estado são CONCENTRAÇÕES
que vazam de um tick para o outro. Entradas 0..162 são fatores de transcrição
grampeados pelo mundo a cada tick (o mundo descreve). Saídas 163..169 são os
sete efetores. Ocultos 200.. começam poucos e só crescem por mutação.

Sem fitness. Herança = cruzamento + mutação no cliente; quem seleciona é o mundo.
"""
from __future__ import annotations

import base64
import gzip
import json
import math
import random

N_IN = 163
N_OUT = 7
OUT0 = N_IN                    # 163
HID0 = 200
MAX_HIDDEN = 24
N_HID_BIRTH = 6
N_REGS_BIRTH = 18

# mutação (taxas do mesmo estatuto que o config-native: mecanismo, não KPI)
P_PERTURB_W = 0.80
P_ADD_REG = 0.15
P_TOGGLE = 0.08
P_PERTURB_DECAY = 0.10
P_ADD_HIDDEN = 0.03
W_SIGMA = 0.4
DECAY_SIGMA = 0.05
DECAY_MIN, DECAY_MAX = 0.05, 0.80


def _tanh(x):
    if x > 8.0:
        return 1.0
    if x < -8.0:
        return -1.0
    return math.tanh(x)


def _clamp_decay(d):
    return max(DECAY_MIN, min(DECAY_MAX, float(d)))


def _hidden_ids(nh):
    return list(range(HID0, HID0 + nh))


def _output_ids():
    return list(range(OUT0, OUT0 + N_OUT))


def _regulated_ids(nh):
    return _hidden_ids(nh) + _output_ids()


def _all_src_ids(nh):
    return list(range(N_IN)) + _regulated_ids(nh)


class Genome:
    """regs: lista [src, dst, w, enabled]. bias/decay: dict id->float só dos regulados."""

    __slots__ = ("nh", "bias", "decay", "regs")

    def __init__(self, nh, bias, decay, regs):
        self.nh = int(nh)
        self.bias = {int(k): float(v) for k, v in bias.items()}
        self.decay = {int(k): _clamp_decay(v) for k, v in decay.items()}
        self.regs = [[int(s), int(d), float(w), bool(en)] for s, d, w, en in regs]

    def to_dict(self):
        return {
            "nh": self.nh,
            "bias": {str(k): round(v, 6) for k, v in self.bias.items()},
            "decay": {str(k): round(v, 6) for k, v in self.decay.items()},
            "regs": [[s, d, round(w, 6), int(en)] for s, d, w, en in self.regs],
        }

    @staticmethod
    def from_dict(d):
        return Genome(d["nh"], d.get("bias") or {}, d.get("decay") or {}, d.get("regs") or [])


class Soup:
    """Concentrações vivas. Um passo = um tick do mundo (sem sub-tick)."""

    def __init__(self, genome: Genome):
        self.g = genome
        self.conc = {i: 0.0 for i in _regulated_ids(genome.nh)}

    def step(self, inputs):
        if len(inputs) != N_IN:
            inputs = (list(inputs) + [0.0] * N_IN)[:N_IN]
        src = dict(self.conc)
        for i, v in enumerate(inputs):
            src[i] = float(v)
        g = self.g
        new = {}
        for i in _regulated_ids(g.nh):
            s = g.bias.get(i, 0.0)
            for a, b, w, en in g.regs:
                if en and b == i:
                    s += w * src.get(a, 0.0)
            produced = _tanh(s)
            dec = g.decay.get(i, 0.25)
            new[i] = (1.0 - dec) * src.get(i, 0.0) + dec * produced
        self.conc = new
        return [self.conc.get(OUT0 + k, 0.0) for k in range(N_OUT)]


def random_genome(seed):
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    nh = N_HID_BIRTH
    bias, decay = {}, {}
    for i in _regulated_ids(nh):
        bias[i] = rng.uniform(-0.3, 0.3)
        decay[i] = rng.uniform(0.15, 0.40)
    regs = []
    srcs = _all_src_ids(nh)
    dsts = _regulated_ids(nh)
    seen = set()
    for _ in range(N_REGS_BIRTH):
        a, b = rng.choice(srcs), rng.choice(dsts)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        regs.append([a, b, rng.uniform(-1.5, 1.5), True])
    return Genome(nh, bias, decay, regs)


def mutate(g: Genome, rng=None):
    rng = rng or random
    g = Genome(g.nh, dict(g.bias), dict(g.decay), [list(r) for r in g.regs])
    if g.regs and rng.random() < P_PERTURB_W:
        r = rng.choice(g.regs)
        r[2] = max(-8.0, min(8.0, r[2] + rng.gauss(0.0, W_SIGMA)))
    if rng.random() < P_ADD_REG:
        srcs = _all_src_ids(g.nh)
        dsts = _regulated_ids(g.nh)
        for _ in range(8):
            a, b = rng.choice(srcs), rng.choice(dsts)
            if a == b:
                continue
            if any(x[0] == a and x[1] == b for x in g.regs):
                continue
            g.regs.append([a, b, rng.uniform(-1.5, 1.5), True])
            break
    if g.regs and rng.random() < P_TOGGLE:
        r = rng.choice(g.regs)
        r[3] = not r[3]
    if rng.random() < P_PERTURB_DECAY and g.decay:
        k = rng.choice(list(g.decay))
        g.decay[k] = _clamp_decay(g.decay[k] + rng.gauss(0.0, DECAY_SIGMA))
        g.bias[k] = max(-4.0, min(4.0, g.bias.get(k, 0.0) + rng.gauss(0.0, 0.1)))
    if g.nh < MAX_HIDDEN and rng.random() < P_ADD_HIDDEN:
        nid = HID0 + g.nh
        g.nh += 1
        g.bias[nid] = rng.uniform(-0.2, 0.2)
        g.decay[nid] = rng.uniform(0.15, 0.40)
        # uma aresta de um src existente para o gene novo, senão ele nasce mudo
        srcs = _all_src_ids(g.nh - 1) or list(range(N_IN))
        g.regs.append([rng.choice(srcs), nid, rng.uniform(-1.0, 1.0), True])
    return g


def crossover(a: Genome, b: Genome, seed):
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    nh = max(a.nh, b.nh)
    bias, decay = {}, {}
    for i in _regulated_ids(nh):
        pa, pb = a.bias.get(i), b.bias.get(i)
        if pa is None:
            bias[i] = pb if pb is not None else 0.0
        elif pb is None:
            bias[i] = pa
        else:
            bias[i] = pa if rng.random() < 0.5 else pb
        da, db = a.decay.get(i), b.decay.get(i)
        if da is None:
            decay[i] = db if db is not None else 0.25
        elif db is None:
            decay[i] = da
        else:
            decay[i] = da if rng.random() < 0.5 else db
    index_a = {(r[0], r[1]): r for r in a.regs}
    index_b = {(r[0], r[1]): r for r in b.regs}
    keys = set(index_a) | set(index_b)
    regs = []
    for k in keys:
        ra, rb = index_a.get(k), index_b.get(k)
        if ra and rb:
            regs.append(list(ra if rng.random() < 0.5 else rb))
        else:
            regs.append(list(ra or rb))
    return Genome(nh, bias, decay, regs)


def complexity(g: Genome):
    """nodes = genes regulados (ocultos+saídas). conns = ligações ENABLED."""
    nodes = g.nh + N_OUT
    conns = sum(1 for r in g.regs if r[3])
    return nodes, conns


def functional_complexity(g: Genome):
    """Alcance para trás a partir das saídas, só enabled — a mesma régua do census.py."""
    adj = {}
    for s, d, _w, en in g.regs:
        if en:
            adj.setdefault(d, []).append(s)
    reach = set(_output_ids())
    stack = list(reach)
    while stack:
        cur = stack.pop()
        for src in adj.get(cur, ()):
            if src not in reach:
                reach.add(src)
                stack.append(src)
    fconns = sum(1 for s, d, _w, en in g.regs if en and d in reach)
    fnodes = sum(1 for i in _regulated_ids(g.nh) if i in reach)
    return fnodes, fconns


def genes_count(g: Genome):
    """§21: tecido carregado = genes regulados + todas as ligações (enabled ou não)."""
    return g.nh + N_OUT + len(g.regs)


def pack(g: Genome) -> str:
    raw = json.dumps(g.to_dict(), separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 6)).decode("ascii")


def unpack(pkt) -> Genome:
    if isinstance(pkt, dict):
        return Genome.from_dict(pkt)
    raw = gzip.decompress(base64.b64decode(pkt))
    return Genome.from_dict(json.loads(raw))

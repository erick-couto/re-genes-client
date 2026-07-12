"""
neat_brain.py — Cérebro NEAT como "pacote" serializável (Fase 2, lado EXECUTOR).

Arquitetura da Fase 2:
  - O MUNDO trata o cérebro como um BLOB OPACO (JSON): ele herda, arquiva no seed
    bank e entrega ao filho no nascimento. O mundo não entende NEAT nenhum.
  - Aqui, no cliente executor, é onde o NEAT acontece de verdade: criar cérebro
    novo, CRESCER a topologia (mutação estrutural — neurônios/ligações novas),
    rodar o forward pass, e serializar pra ida/volta com o mundo.

Sem população e sem função de fitness: o executor só EXECUTA e aplica a HERANÇA
(mutação no nascimento). Quem seleciona é o mundo (sobreviver e reproduzir).

Reusa o mesmo neat-python (fork 1.1.0) e o mesmo config-feedforward do client_neat,
pra o "cérebro nativo" ser do mesmo sangue que a espécie NEAT já existente.
"""
import base64
import gzip
import json
import os

import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

_CONFIG = None
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "client_neat", "config-feedforward")


def load_config(path: str = None) -> neat.Config:
    """Carrega (e memoiza) o config NEAT. Define num_inputs/outputs e taxas de mutação."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            path or _DEFAULT_CONFIG,
        )
        # O fork exige um InnovationTracker no genome_config para criar/crescer genomas
        # (garante ids consistentes de neurônios/ligações novas). Um por executor basta.
        if getattr(_CONFIG.genome_config, "innovation_tracker", None) is None:
            from neat.innovation import InnovationTracker
            _CONFIG.genome_config.innovation_tracker = InnovationTracker()
    return _CONFIG


def random_genome(key: int = 0):
    """Cérebro da 'sopa primordial': genoma novo conforme o config (topologia inicial)."""
    cfg = load_config()
    g = neat.DefaultGenome(key)
    g.configure_new(cfg.genome_config)
    return g


def mutate(genome):
    """Herança com variação: mutação estrutural (cresce) + de pesos, in-place. Retorna o genoma."""
    cfg = load_config()
    genome.mutate(cfg.genome_config)
    return genome


def build_net(genome):
    """Rede executável (forward pass) a partir do genoma."""
    cfg = load_config()
    return neat.nn.FeedForwardNetwork.create(genome, cfg)


def complexity(genome):
    """(n_neuronios, n_ligacoes_ativas) — pra medir o crescimento do cérebro."""
    active = sum(1 for c in genome.connections.values() if c.enabled)
    return (len(genome.nodes), active)


# --- SERIALIZAÇÃO: genoma <-> dict JSON (o "pacote" que trafega pro mundo) ---

def to_dict(genome) -> dict:
    """Genoma NEAT -> dict JSON-serializável (o blob opaco que o mundo guarda)."""
    return {
        "key": genome.key,
        # nó: [bias, response, activation, aggregation]
        "nodes": {str(nid): [ng.bias, ng.response, ng.activation, ng.aggregation]
                  for nid, ng in genome.nodes.items()},
        # ligação: [in, out, weight, enabled, innovation]
        "conns": [[k[0], k[1], cg.weight, bool(cg.enabled), getattr(cg, "innovation", 0)]
                  for k, cg in genome.connections.items()],
    }


def from_dict(d: dict):
    """dict JSON -> genoma NEAT (reconstrói pra rodar/mutar)."""
    g = neat.DefaultGenome(d.get("key", 0))
    g.nodes = {}
    for nid_s, (bias, resp, act, agg) in d["nodes"].items():
        nid = int(nid_s)
        ng = DefaultNodeGene(nid)
        ng.bias, ng.response, ng.activation, ng.aggregation = bias, resp, act, agg
        g.nodes[nid] = ng
    g.connections = {}
    for row in d["conns"]:
        i, o, w, en = row[0], row[1], row[2], row[3]
        innov = row[4] if len(row) > 4 else 0
        cg = DefaultConnectionGene((int(i), int(o)), innov)
        cg.weight, cg.enabled = w, bool(en)
        g.connections[(int(i), int(o))] = cg
    return g


# --- PACOTE COMPACTO: o que trafega/armazena (base64(gzip(json))) ---
# O mundo guarda/entrega essa STRING opaca; só o executor comprime/descomprime.
# ~24 KB de JSON -> ~4-6 KB. String ASCII, cabe em JSON (WELCOME/report/snapshot).

def pack(genome) -> str:
    """Genoma -> pacote compacto (string base64 de gzip(json))."""
    raw = json.dumps(to_dict(genome), separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 6)).decode("ascii")


def unpack(pkt: str):
    """Pacote compacto -> genoma. Aceita dict cru também (retrocompat com JSON não comprimido)."""
    if isinstance(pkt, dict):        # tolera blobs antigos (dict JSON puro)
        return from_dict(pkt)
    raw = gzip.decompress(base64.b64decode(pkt))
    return from_dict(json.loads(raw))


# --- ARQUIVO .brain: exportar/propagar um campeão (magic + gzip(json)) ---
_BRAIN_MAGIC = b"RGB1"  # re-genes brain, formato v1

def save_brain(genome, path: str):
    """Salva um cérebro num arquivo .brain (magic + gzip(json), compressão máxima)."""
    raw = json.dumps(to_dict(genome), separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(_BRAIN_MAGIC)
        f.write(gzip.compress(raw, 9))


def load_brain(path: str):
    """Carrega um arquivo .brain -> genoma."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != _BRAIN_MAGIC:
        raise ValueError("arquivo .brain inválido (magic ausente)")
    return from_dict(json.loads(gzip.decompress(data[4:])))

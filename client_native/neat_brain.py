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
import hashlib
import json
import os
import types
from random import choice

import neat
from neat.graphs import creates_cycle
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.innovation import InnovationTracker

# =============================================================================
# IDENTIDADE ESTRUTURAL GLOBAL E DETERMINISTICA (fix do crossover distribuido)
# -----------------------------------------------------------------------------
# Problema: o fork numera INOVACAO (InnovationTracker.global_counter) e ID DE NO
# (get_new_node_key = contador) POR PROCESSO. Cada `python host.py` comeca do zero,
# mas o brain_bank do mundo PERSISTE entre restarts/maquinas. Logo, a mesma inovacao
# (ou id de no) significa conexoes diferentes em linhagens diferentes -> o crossover
# nao alinha, descarta ~metade dos genes por acasalamento e emite "Innovation collision".
#
# Fix: ancorar a identidade na ESTRUTURA, nao em quem/quando criou. A inovacao de uma
# conexao vira funcao estavel de (in,out); o id de um no novo vira funcao estavel da
# conexao dividida. Assim a mesma mutacao estrutural recebe a MESMA identidade em todo
# processo/maquina/restart -> o crossover alinha de verdade (a diversidade que ele deve
# misturar), sem colisao. E numeracao deterministica, sem estado global compartilhado.
# =============================================================================
_NODE_ID_BASE = 100_000  # nos de MUTACAO ficam >= isto; nunca colidem com saidas(0..6),
                         # ocultos semeados(7..) nem inputs(negativos).


def _stable_hash(text: str, nbytes: int) -> int:
    """Hash estavel entre processos/versoes (sha256; NAO usa hash() nativo, que e aleatorizado)."""
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:nbytes], "big")


def _det_innovation(in_node: int, out_node: int) -> int:
    """Inovacao = funcao estavel de (in,out). Mesma conexao -> mesmo numero, em todo lugar."""
    return _stable_hash(f"conn:{in_node}:{out_node}", 8)  # 64 bits


def _det_node_id(in_node: int, out_node: int) -> int:
    """Id do no novo = funcao estavel da conexao dividida (in,out). Mesmo split -> mesmo id."""
    return _NODE_ID_BASE + _stable_hash(f"node:{in_node}:{out_node}", 6)  # >= 100000


class DeterministicInnovationTracker(InnovationTracker):
    """Drop-in do tracker do fork: ignora o contador e devolve inovacao deterministica por (in,out)."""

    def get_innovation_number(self, input_node, output_node, mutation_type="add_connection"):
        return _det_innovation(input_node, output_node)


def _det_get_new_node_key(self, node_dict):
    """Ids DETERMINISTICOS p/ ocultos SEMEADOS (configure_new): num_outputs + (quantos ja existem
    na faixa fixa). Todo genoma fresco recebe os mesmos ids 7..(7+num_hidden-1), sem contador
    compartilhado -> ocultos semeados alinham no crossover. Nos de MUTACAO nao passam por aqui
    (ver _det_mutate_add_node)."""
    seeded = [k for k in node_dict if self.num_outputs <= k < _NODE_ID_BASE]
    return self.num_outputs + len(seeded)


def _det_mutate_add_node(self, config):
    """mutate_add_node com id de no DETERMINISTICO (= _det_node_id da conexao dividida). Copia fiel
    do fork, exceto: (a) id vem da estrutura, nao do contador; (b) divide so conexao ATIVADA (evita
    re-split de desabilitada, que geraria id ja existente); (c) se o split ja existe no genoma, pula."""
    if not self.connections:
        if config.check_structural_mutation_surer():
            self.mutate_add_connection(config)
        return
    enabled = [cg for cg in self.connections.values() if cg.enabled]
    if not enabled:
        return
    conn_to_split = choice(enabled)
    i, o = conn_to_split.key
    new_node_id = _det_node_id(i, o)
    if new_node_id in self.nodes:
        return  # esse split ja foi feito neste genoma; nao duplica
    ng = self.create_node(config, new_node_id)
    if hasattr(ng, "bias"):
        ng.bias = 0.0
    self.nodes[new_node_id] = ng
    conn_to_split.enabled = False
    in_innov = config.innovation_tracker.get_innovation_number(i, new_node_id, "add_node_in")
    out_innov = config.innovation_tracker.get_innovation_number(new_node_id, o, "add_node_out")
    self.add_connection(config, i, new_node_id, 1.0, True, innovation=in_innov)
    self.add_connection(config, new_node_id, o, conn_to_split.weight, True, innovation=out_innov)


# --- CROSSOVER SIMETRICO: a clausula do FITNESS IGUAL que o neat-python nao implementa ---
# Stanley & Miikkulainen (2002), p.108: "Genes are randomly chosen from either parent at matching
# genes, whereas all excess or disjoint genes are always included from the more fit parent." E,
# logo em seguida: "in the case of EQUAL fitness, disjoint and excess genes are inherited RANDOMLY."
#
# A biblioteca so implementa a PRIMEIRA frase:
#     if genome1.fitness > genome2.fitness: parent1, parent2 = genome1, genome2
#     else:                                 parent1, parent2 = genome2, genome1
#     ...
#     # Note: genes only in parent2 (less fit) are not inherited
# O EMPATE cai no `else` e vira "g2 e o mais apto": os genes exclusivos de g1 sao DESCARTADOS em
# silencio. Nao e bug da lib — e a clausula que falta.
#
# E o nosso mundo IGUALA o fitness de proposito (a selecao e do MUNDO, nao do executor), entao
# TODO acasalamento caia nesse caso. Com 100% de reproducao sexuada (medido: 58/58), cada
# nascimento jogava fora os genes exclusivos de um dos pais. Foi a EROSAO que perseguimos a
# sessao inteira (741->36, 113->27, 113->15): nao era node_delete (removido), nao era a
# numeracao de inovacao (consertada) — era isto. Medido: filho ficava 100% igual a um pai e 0%
# ao outro, e 35% dos filhos nasciam com menos conns que o pai mais magro.
def _sym_configure_crossover(self, genome1, genome2, config):
    empate = abs((genome1.fitness or 0.0) - (genome2.fitness or 0.0)) < 1e-9
    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    p1 = {cg.innovation: cg for cg in parent1.connections.values()}
    p2 = {cg.innovation: cg for cg in parent2.connections.values()}

    for inn in set(p1) | set(p2):
        cg1, cg2 = p1.get(inn), p2.get(inn)
        if cg1 is not None and cg2 is not None:
            # homologo: sorteia dos dois (ja era assim). Colisao de inovacao -> trata como disjunto.
            gene = cg1.copy() if cg1.key != cg2.key else cg1.crossover(cg2)
        elif cg1 is not None:
            gene = cg1.copy()                 # disjunto do mais apto: sempre entra
        else:
            # disjunto do MENOS apto. Canonico: so entra se houver EMPATE, e ai por SORTEIO.
            # Sem isto, este gene morria SEMPRE — que era exatamente a erosao.
            if not (empate and choice([True, False])):
                continue
            gene = cg2.copy()
        if config.feed_forward and creates_cycle(list(self.connections), gene.key):
            continue
        self.connections[gene.key] = gene

    for key, ng1 in parent1.nodes.items():
        ng2 = parent2.nodes.get(key)
        self.nodes[key] = ng1.copy() if ng2 is None else ng1.crossover(ng2)

    # nos exigidos pelas conexoes herdadas do outro pai (senao o genoma nasce quebrado:
    # conexao apontando pra neuronio inexistente)
    for (i, o) in list(self.connections):
        for k in (i, o):
            if k not in self.nodes and k in parent2.nodes:
                self.nodes[k] = parent2.nodes[k].copy()


def _install_symmetric_crossover(cfg) -> None:
    neat.DefaultGenome.configure_crossover = _sym_configure_crossover


def _install_deterministic_identity(cfg: neat.Config) -> None:
    """Instala a identidade estrutural deterministica no config (tracker + id de no semeado + add_node)."""
    gc = cfg.genome_config
    gc.innovation_tracker = DeterministicInnovationTracker()
    gc.get_new_node_key = types.MethodType(_det_get_new_node_key, gc)  # so p/ este config
    neat.DefaultGenome.mutate_add_node = _det_mutate_add_node          # patch de classe (so amebas nativas)


_CONFIG = None
# Config ISOLADO do nativo (161 entradas v3 EGOCENTRICO: bias+energy+stomach+endorfina+
# marca-passo(sin,cos) + 5 canais x 31 do cone frontal). NAO usa o config-feedforward do
# client_neat (104 entradas) — as duas especies divergiram de proposito.
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config-native")


def load_config(path: str = None) -> neat.Config:
    """Carrega (e memoiza) o config NEAT. Define num_inputs/outputs e taxas de mutação."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            path or _DEFAULT_CONFIG,
        )
        # Identidade estrutural GLOBAL e DETERMINISTICA (inovacao + id de no). Substitui o
        # tracker por-processo do fork -> genomas de qualquer processo/maquina/restart alinham
        # no crossover. Ver bloco no topo do modulo.
        _install_deterministic_identity(_CONFIG)
        _install_symmetric_crossover(_CONFIG)   # clausula do fitness igual (Stanley p.108)
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


def crossover(g1, g2, key: int = 0):
    """
    Cruzamento sexual NEAT — MISTURA linhagens de clientes/máquinas diferentes (a fonte de
    diversidade no mundo distribuído). Recombinação alinhada por número de inovação; genes
    disjuntos/excedentes vêm do pai mais apto. Fitness igualada (a seleção é do MUNDO) ->
    cruzamento SIMÉTRICO. Retorna o filho ainda SEM mutar (mute depois).

    BUG ESTRUTURAL ABERTO: inovação (InnovationTracker.global_counter) e id de neurônio
    (get_new_node_key = max+1) são atribuídos POR PROCESSO, mas os genomas circulam entre
    clientes. Linhagens de máquinas diferentes não alinham -> genes viram disjuntos, o filho
    descarta ~metade a cada acasalamento (erosão de conexões) e emite "Innovation number
    collision". Fix correto = id de nó/inovação GLOBAL e DETERMINÍSTICO (derivado da estrutura,
    ex.: id do nó = hash da conexão dividida), igual em todo cliente. NÃO remover o crossover.
    """
    cfg = load_config()
    g1.fitness = g1.fitness if getattr(g1, "fitness", None) is not None else 1.0
    g2.fitness = g2.fitness if getattr(g2, "fitness", None) is not None else 1.0
    child = neat.DefaultGenome(key)
    child.configure_crossover(g1, g2, cfg.genome_config)
    return child


def build_net(genome):
    """Rede executável (forward pass) a partir do genoma."""
    cfg = load_config()
    return neat.nn.FeedForwardNetwork.create(genome, cfg)


def complexity(genome):
    """(n_neuronios, n_ligacoes_ativas) — pra medir o crescimento do cérebro."""
    active = sum(1 for c in genome.connections.values() if c.enabled)
    return (len(genome.nodes), active)


def functional_complexity(genome):
    """(n_neuronios, n_ligacoes) FUNCIONAIS — o cérebro de verdade, não o genoma.

    Mesmo critério do FeedForwardNetwork.create (neat.graphs.required_for_output): um nó só
    computa se tem caminho até uma SAÍDA por conexões HABILITADAS. O resto é material morto
    (split de conexão desligada, toggle do enabled_mutate, órfãos de crossover) — existe no
    genoma, é carregado e copiado, mas não pensa. Medido em produção (07/2026): 99% dos nós
    ocultos do brain_bank eram mortos; a telemetria de `complexity` lia "cérebro crescendo"
    (21->414 nós) quando o funcional ficou em ~6-14 ligações. Desde então a telemetria e o
    mundo reportam as DUAS medidas: genoma (custa DNA, §21) e cérebro real (observabilidade).
    """
    cfg = load_config()
    outputs = set(cfg.genome_config.output_keys)
    adj = {}
    for (i, o), cg in genome.connections.items():
        if cg.enabled:
            adj.setdefault(o, []).append(i)
    reach = set(outputs)
    stack = list(outputs)
    while stack:
        cur = stack.pop()
        for src in adj.get(cur, ()):
            if src not in reach:
                reach.add(src)
                stack.append(src)
    fnodes = len(set(genome.nodes) & reach)
    fconns = sum(1 for (i, o), cg in genome.connections.items() if cg.enabled and o in reach)
    return (fnodes, fconns)


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

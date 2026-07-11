"""
Espécie NEAT (neuroevolução de topologia) — portada para o Species Protocol v1.

Reusa o ContinuousPopulation (GA steady-state) do client_neat.py legado, mas a
conexão/loop agora vêm do SDK (regenes_agent.BaseAgent). Assim o NEAT entra na
arena no mesmo contrato que Q-table, ES, etc.

Mantém 79 entradas (bias + energy + stomach + endorfina + 3×25 de visão) e a
endorfina como sensor interno, para continuidade com checkpoints existentes.
"""
import os

import neat

from regenes_agent import BaseAgent
import client_neat as _legacy
from client_neat import ContinuousPopulation  # reusa o GA steady-state legado

# O checkpoint legado foi picklado com client_neat.py rodando como __main__, então
# referencia __main__.PicklableCount. Ao importar como módulo, injetamos a classe no
# __main__ atual para o unpickle resolver — de qualquer ponto de entrada (run.py, teste).
import sys as _sys
_main = _sys.modules.get("__main__")
if _main is not None and not hasattr(_main, "PicklableCount"):
    _main.PicklableCount = _legacy.PicklableCount

VC = 4  # centro do 9x9 (raio 4)

# pesos da fitness V7 (sobrevivência + forrageamento)
FOOD_REWARD, ENERGY_REWARD, SURVIVAL_REWARD, EXPLORE_REWARD = 200.0, 1.0, 1.0, 0.25


class NeatAgent(BaseAgent):
    species = "NEAT_Evo"
    paradigm = "neuroevolution_topology"

    def __init__(self, gid, genome, pop):
        super().__init__()
        self.gid = gid
        self.genome = genome
        self.pop = pop
        self.net = neat.nn.FeedForwardNetwork.create(genome, pop.config)
        self.ticks = 0
        self.food = 0
        self.energy_gained = 0.0
        self.last_energy = None
        self.visited = set()
        self.last_pos = None
        self.endorphin = 50.0

    def _inputs(self, vision, energy, stomach):
        if not vision or len(vision[0]) < 9:
            return [0.0] * 79
        inp = [1.0,
               self.energy_norm(energy),
               min(stomach, self.stomach_size) / self.stomach_size,
               self.endorphin / 100.0]
        for ch in (0, 1, 2):  # walls, scent, enemies
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    val = vision[ch][VC + dy][VC + dx]
                    inp.append(1.0 if (ch == 0 and val > 0) else val)
        return inp

    def decide(self, obs) -> int:
        energy = obs.get("energy", 0)
        # endorfina: decaimento + estresse de fome (sensor interno)
        self.endorphin -= 0.2
        if energy < self.stomach_size * 0.5:
            self.endorphin -= 2.0
        self.endorphin = max(0.0, min(100.0, self.endorphin))
        outputs = self.net.activate(self._inputs(obs.get("vision"), energy, obs.get("stomach", 0)))
        self.ticks += 1
        return int(max(range(len(outputs)), key=lambda i: outputs[i]))

    def on_update(self, upd):
        e = upd.get("energy", 0)
        if self.last_energy is not None and e > self.last_energy:
            self.energy_gained += (e - self.last_energy)
            self.food += 1
            self.endorphin = min(100.0, self.endorphin + 100.0)
        self.last_energy = e
        pos = (upd.get("x"), upd.get("y"))
        if pos[0] is not None and pos != self.last_pos:
            self.endorphin = min(100.0, self.endorphin + 0.3)
            if pos not in self.visited:
                self.visited.add(pos)
                self.endorphin = min(100.0, self.endorphin + 5.0)
            self.last_pos = pos

    def on_death(self, final):
        fitness = (self.food * FOOD_REWARD + self.energy_gained * ENERGY_REWARD
                   + self.ticks * SURVIVAL_REWARD + len(self.visited) * EXPLORE_REWARD)
        self.pop.report_death(self.gid, fitness)
        _log_perf(self.gid, fitness, self.ticks, self.food)
        _maybe_save(self.pop)


# --- persistência / telemetria (mesmos arquivos do legado) ---
_deaths = {"n": 0}


def _maybe_save(pop, every=50):
    _deaths["n"] += 1
    if _deaths["n"] % every == 0:
        pop.save_checkpoint()


def _log_perf(gid, fitness, ticks, food):
    f = os.path.join(os.path.dirname(__file__), "neat_performance.csv")
    new = not os.path.exists(f)
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write("id,fitness,ticks,food_eaten\n")
        fh.write(f"{gid},{fitness:.2f},{ticks},{food}\n")


def make_factory():
    """Cria a população compartilhada e devolve um factory que entrega um genoma
    novo a cada vida (e recebe a fitness de volta no on_death)."""
    base = os.path.dirname(__file__)
    config_path = os.path.join(base, "config-feedforward")
    pop = ContinuousPopulation(config_path, checkpoint_file=os.path.join(base, "neat-checkpoint-continuous-auto"))

    def factory():
        gid, genome = pop.get_genome()
        return NeatAgent(gid, genome, pop)

    return factory, pop

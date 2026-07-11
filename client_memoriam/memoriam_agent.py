"""
Espécie Memoriam (Q-learning tabular) — portada para o Species Protocol v1.

Reusa BrainManager/MemoriamBrain do client_memoriam.py legado (Q-tables por
fenótipo, compartilhadas entre amebas da mesma espécie). A conexão/loop vêm do
SDK. Aprendizado é Lamarckiano: a experiência de cada vida é escrita de volta na
Q-table compartilhada (self_learns=True).
"""
import os

from regenes_agent import BaseAgent
from client_memoriam import BrainManager  # reusa a lógica tabular legada

VC = 4  # centro do 9x9
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]  # índice == action_spec do servidor


class MemoriamAgent(BaseAgent):
    species = "Memoriam"
    paradigm = "rl_tabular"
    self_learns = True

    # gerenciador de cérebros COMPARTILHADO entre todas as instâncias desta espécie
    manager = BrainManager()

    def __init__(self):
        super().__init__()
        self.brain = None
        self.last_state = None
        self.last_action = None   # label
        self.last_energy = 100

    def on_welcome(self, welcome):
        super().on_welcome(welcome)
        # o fenótipo (ex.: "Giant Slow (Offspring...)") define qual Q-table usar
        self.brain = MemoriamAgent.manager.get_brain(welcome.get("species", "Unknown"))

    def _state(self, vision, energy):
        if not vision:
            return "BLIND"
        parts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                wall = 1 if vision[0][VC + dy][VC + dx] > 0 else 0
                sc = vision[1][VC + dy][VC + dx]
                lvl = 2 if sc > 0.5 else (1 if sc > 0.05 else 0)
                parts.append(f"{wall}{lvl}")
        e = "CRIT" if energy < 20 else ("LOW" if energy < 50 else "OK")
        return "".join(parts) + "_" + e

    def decide(self, obs) -> int:
        energy = obs.get("energy", self.last_energy)
        state = self._state(obs.get("vision"), energy)
        # aprende do passo anterior (reward por delta de energia)
        if self.last_state is not None and self.last_action is not None:
            delta = energy - self.last_energy
            reward = 50 if delta > 0 else (-0.1 if delta == 0 else -1)
            self.brain.update(self.last_state, self.last_action, reward, state)
        action = self.brain.get_action(state)  # label, epsilon-greedy
        self.last_state, self.last_action, self.last_energy = state, action, energy
        return ACTIONS.index(action) if action in ACTIONS else 4

    def on_death(self, final):
        if self.last_state is not None and self.last_action is not None:
            self.brain.update(self.last_state, self.last_action, -100, "DEATH")
        _maybe_save()


_deaths = {"n": 0}


def _maybe_save(every=40):
    _deaths["n"] += 1
    if _deaths["n"] % every == 0:
        for b in MemoriamAgent.manager.brains.values():
            b.decay_epsilon()
        MemoriamAgent.manager.save_all()


def make_factory():
    return MemoriamAgent  # o próprio construtor é o factory (cérebros são compartilhados)

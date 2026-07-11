"""
Espécie ES (Evolution Strategies) — OpenAI-ES sobre uma MLP de TOPOLOGIA FIXA.

É o par científico do NEAT: mesma família (neuroevolução), mas os pesos de uma
rede fixa evoluem por perturbação gaussiana + gradiente estimado, sem crescer
topologia. A comparação NEAT vs ES responde: "crescer a arquitetura ajuda mesmo?"

Só numpy (sem GPU). Uma versão com torch/GPU é trivial depois, se quiser um
modelo mais parrudo — o contrato de espécie é agnóstico de custo.
"""
import os

import numpy as np

from regenes_agent import BaseAgent

VC = 4
# v2 (predação): 102 entradas = energy + stomach + 4 canais x 25 (obstáculos, cheiro,
# inimigo, perigo); 9 saídas = 4 move + stay + 4 attack. Mudar => reset do theta.
I, H, O = 102, 16, 9          # entradas, ocultas, ações
SIGMA, ALPHA, BATCH = 0.1, 0.05, 32
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "es_theta_v2.npy")  # v2 = predação

FOOD_REWARD, ENERGY_REWARD, SURVIVAL_REWARD = 200.0, 1.0, 1.0


class ESManager:
    """Mantém o vetor de pesos central (theta) e faz o update ES por lotes de mortes."""

    def __init__(self):
        self.dim = H * I + H + O * H + O
        self.theta = None
        if os.path.exists(WEIGHTS_FILE):
            try:
                t = np.load(WEIGHTS_FILE)
                if t.shape == (self.dim,):     # ignora theta de outra versão (shape errado)
                    self.theta = t.astype(np.float32)
                    print(f"📂 ES: theta carregado ({self.dim} pesos)")
                else:
                    print(f"⚠️ ES: theta salvo shape {t.shape} != ({self.dim},). Começando do zero.")
            except Exception as e:
                print(f"⚠️ ES: falha ao carregar theta ({e}). Começando do zero.")
        if self.theta is None:
            self.theta = np.random.randn(self.dim).astype(np.float32) * 0.1
            print(f"🌱 ES: theta novo ({self.dim} pesos)")
        self.samples = []   # (eps, fitness)
        self.gen = 0

    def sample(self):
        eps = np.random.randn(self.dim).astype(np.float32)
        return self.theta + SIGMA * eps, eps

    def unflatten(self, w):
        i = 0
        W1 = w[i:i + H * I].reshape(H, I); i += H * I
        b1 = w[i:i + H]; i += H
        W2 = w[i:i + O * H].reshape(O, H); i += O * H
        b2 = w[i:i + O]
        return W1, b1, W2, b2

    def report(self, eps, fitness):
        self.samples.append((eps, fitness))
        if len(self.samples) >= BATCH:
            self._update()

    def _update(self):
        F = np.array([f for _, f in self.samples], dtype=np.float32)
        E = np.stack([e for e, _ in self.samples])
        # rank-normalization (robusto a outliers de fitness)
        ranks = F.argsort().argsort().astype(np.float32)
        A = ranks / max(1, len(F) - 1) - 0.5
        grad = (A[:, None] * E).sum(axis=0) / (len(F) * SIGMA)
        self.theta = self.theta + ALPHA * grad
        self.gen += 1
        self.samples = []
        np.save(WEIGHTS_FILE, self.theta)
        best = F.max()
        print(f"📈 [ES gen {self.gen}] update aplicado | fitness lote: média={F.mean():.0f} max={best:.0f}")


class ESAgent(BaseAgent):
    species = "ES_v1"
    paradigm = "neuroevolution_weights"

    manager = ESManager()

    def __init__(self):
        super().__init__()
        self.w, self.eps = ESAgent.manager.sample()
        self.W1, self.b1, self.W2, self.b2 = ESAgent.manager.unflatten(self.w)
        self.ticks = 0
        self.food = 0
        self.energy_gained = 0.0
        self.last_energy = None

    def _inputs(self, vision, energy, stomach):
        x = np.zeros(I, dtype=np.float32)
        if not vision or len(vision) < 4 or len(vision[0]) < 9:
            return x
        x[0] = self.energy_norm(energy)
        x[1] = min(stomach, self.stomach_size) / self.stomach_size
        k = 2
        for ch in (0, 1, 2, 3):  # obstáculos, cheiro, inimigo, perigo
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    v = vision[ch][VC + dy][VC + dx]
                    x[k] = 1.0 if (ch == 0 and v > 0) else v
                    k += 1
        return x

    def decide(self, obs) -> int:
        x = self._inputs(obs.get("vision"), obs.get("energy", 0), obs.get("stomach", 0))
        h = np.tanh(self.W1 @ x + self.b1)
        o = self.W2 @ h + self.b2
        self.ticks += 1
        return int(o.argmax())

    def on_update(self, upd):
        e = upd.get("energy", 0)
        if self.last_energy is not None and e > self.last_energy:
            self.energy_gained += (e - self.last_energy)
            self.food += 1
        self.last_energy = e

    def on_death(self, final):
        fitness = self.food * FOOD_REWARD + self.energy_gained * ENERGY_REWARD + self.ticks * SURVIVAL_REWARD
        ESAgent.manager.report(self.eps, fitness)
        _log_perf(fitness, self.ticks, self.food)


def _log_perf(fitness, ticks, food):
    f = os.path.join(os.path.dirname(__file__), "es_performance_v2.csv")
    new = not os.path.exists(f)
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write("gen,fitness,ticks,food_eaten\n")
        fh.write(f"{ESAgent.manager.gen},{fitness:.2f},{ticks},{food}\n")


def make_factory():
    return ESAgent

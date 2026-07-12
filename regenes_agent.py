"""
re-genes — SDK de Agente (Species Protocol v1)

A ARENA de abordagens de ML. O servidor descreve o mundo (observation_spec /
action_spec no WELCOME); o cliente traz a inteligência. Qualquer controlador
— NEAT, Q-table, DQN, PPO, ES, heurística, humano — pluga aqui subclassando
BaseAgent e implementando UM método: decide(obs) -> índice de ação.

Plugar uma abordagem nova:

    class MinhaIA(BaseAgent):
        species = "MinhaIA"
        paradigm = "deep_rl"
        self_learns = True                 # aprende durante a vida (Lamarckiano)
        def decide(self, obs) -> int:
            return self.model.act(obs)     # obs = {vision, energy, stomach, tick}
        def on_update(self, upd): ...       # feedback de energia/posição (p/ aprender)
        def on_death(self, final): ...      # fim de vida (treinar/salvar)

    run(MinhaIA, n=8)                       # sobe 8 amebas dessa espécie

O 'factory' pode fechar sobre um modelo COMPARTILHADO (ex.: uma Q-table ou uma
rede DQN única) para aprendizado que persiste entre amebas — ou criar instâncias
independentes para abordagens sem estado. Essa é a alavanca de extensibilidade.
"""
import asyncio
import json
import os
import random
import ssl
import sys

import websockets

# wss:// = produção; ws://127.0.0.1:8123 = local (via env REGENES_SERVER).
SERVER_URL = os.getenv("REGENES_SERVER", "wss://re-genes.is")


def _make_ssl_context():
    """Tolera o MITM local do Avast (VERIFY_X509_STRICT reprova o root dele).
    Se ele servir cert vencido, rode com REGENES_INSECURE_TLS=1."""
    if os.getenv("REGENES_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("⚠️  REGENES_INSECURE_TLS=1 -> verificação de certificado DESLIGADA")
        return ctx
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


SSL_CONTEXT = _make_ssl_context() if SERVER_URL.startswith("wss") else None


class BaseAgent:
    """Contrato universal de espécie. Subclasse e implemente decide()."""

    species = "Base"
    paradigm = "unknown"
    wants_brain = False   # capability: pedir cérebro herdado ao servidor (Fase 2, futuro)
    self_learns = False   # capability: aprende durante a vida (informativo p/ o servidor)

    def __init__(self):
        self.body = {}
        self.action_spec = None
        self.obs_spec = None
        self.stomach_size = 200.0
        self.my_id = None
        self.brain = None

    # --- Ciclo de vida (hooks opcionais) ---
    def on_welcome(self, welcome: dict):
        """Chamado uma vez ao nascer. Guarda corpo e specs; adapte-se aqui."""
        self.my_id = welcome.get("id")
        self.body = welcome.get("body") or welcome.get("stats") or {}
        self.stomach_size = self.body.get("stomach_size", 200.0)
        self.action_spec = welcome.get("action_spec")
        self.obs_spec = welcome.get("observation_spec")
        self.brain = welcome.get("brain")

    def decide(self, obs: dict) -> int:
        """OBRIGATÓRIO. Recebe obs={vision, energy, stomach, tick} e devolve o
        ÍNDICE da ação (0=UP,1=DOWN,2=LEFT,3=RIGHT,4=STAY por padrão)."""
        raise NotImplementedError

    def on_update(self, update: dict):
        """Opcional. Feedback do servidor após cada ação: {alive, energy, x, y}.
        É aqui que um aprendiz calcula reward e atualiza o modelo."""

    def on_death(self, final: dict):
        """Opcional. Fim de vida — treinar em lote, salvar checkpoint, etc."""

    # --- utilidades ---
    def n_actions(self) -> int:
        if self.action_spec:
            return len(self.action_spec["commands"])
        return 5

    def energy_norm(self, energy: float) -> float:
        return min(energy, self.stomach_size) / self.stomach_size if self.stomach_size else 0.0


# =========================================================
#  Runtime: conexão, handshake e loop universal
# =========================================================
async def _run_one(agent: BaseAgent):
    q = (f"/ws/join?species={agent.species}&paradigm={agent.paradigm}"
         f"&wants_brain={int(agent.wants_brain)}&self_learns={int(agent.self_learns)}")
    url = SERVER_URL + q
    ssl_ctx = SSL_CONTEXT if url.startswith("wss") else None
    async with websockets.connect(url, ssl=ssl_ctx) as ws:
        welcome = json.loads(await ws.recv())
        agent.on_welcome(welcome)
        commands = (agent.action_spec or {}).get("commands") or [
            {"wire": {"action": "move", "direction": d}} for d in ("UP", "DOWN", "LEFT", "RIGHT")
        ] + [{"wire": {"action": "stay"}}]

        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            if mtype == "UPDATE":
                agent.on_update(msg)
                if not msg.get("alive", True):
                    agent.on_death(msg)
                    return
            elif mtype == "TICK":
                obs = {
                    "vision": msg.get("vision"),
                    "energy": msg.get("energy", 0),
                    "stomach": msg.get("stomach", 0),
                    "tick": msg.get("tick"),
                }
                idx = agent.decide(obs)
                idx = max(0, min(int(idx), len(commands) - 1))
                await ws.send(json.dumps(commands[idx]["wire"]))


async def run_swarm(factory, n: int):
    """Mantém n amebas vivas dessa espécie; respawna quando morrem.
    'factory' é chamado a cada nova vida (pode fechar sobre um modelo compartilhado)."""
    tasks = set()
    errors = {"count": 0}

    async def spawn():
        agent = None
        try:
            agent = factory()          # DENTRO do try: se o factory (get_genome) lançar,
            await _run_one(agent)       # a gente captura em vez de matar os nascimentos calado.
        except Exception as e:
            errors["count"] += 1
            if errors["count"] <= 5 or errors["count"] % 100 == 0:
                sp = getattr(agent, "species", "?")
                print(f"⚠️  erro #{errors['count']} ({sp}): {type(e).__name__}: {e}")
            await asyncio.sleep(0.25)    # evita loop apertado de falha (não pega 100% de CPU)

    print(f"🚀 Arena: subindo {n}x {factory().species} contra {SERVER_URL}")
    while True:
        while len(tasks) < n:
            t = asyncio.create_task(spawn())
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            await asyncio.sleep(0.1)
        if tasks:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        else:
            await asyncio.sleep(1)


def run(factory, n: int = 8):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_swarm(factory, n))
    except KeyboardInterrupt:
        print("\n👋 encerrando.")


# =========================================================
#  Espécies de REFERÊNCIA (baselines da arena)
# =========================================================
class RandomAgent(BaseAgent):
    """Ruído puro. O piso da arena."""
    species = "Random"
    paradigm = "baseline_random"

    def decide(self, obs) -> int:
        return random.randint(0, self.n_actions() - 1)


class GreedyAgent(BaseAgent):
    """Heurística: anda para o vizinho de maior cheiro (grupo de CONTROLE).
    Se seu RL não bater isto, ele não está aprendendo nada."""
    species = "Greedy"
    paradigm = "baseline_heuristic"

    def decide(self, obs) -> int:
        v = obs.get("vision")
        if not v:
            return 4
        walls, scent = v[0], v[1]
        c = 4  # centro do 9x9
        best_idx, best_val = 4, 0.0  # default: STAY
        for idx, (dy, dx) in [(0, (-1, 0)), (1, (1, 0)), (2, (0, -1)), (3, (0, 1))]:
            y, x = c + dy, c + dx
            if walls[y][x] > 0:
                continue
            if scent[y][x] > best_val:
                best_val, best_idx = scent[y][x], idx
        if best_val > 0.0:
            return best_idx
        return random.randint(0, 3)  # sem cheiro à vista: perambula


_SPECIES = {"random": RandomAgent, "greedy": GreedyAgent}

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "greedy"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    factory = _SPECIES.get(kind)
    if factory is None:
        print(f"espécie desconhecida '{kind}'. opções: {list(_SPECIES)}")
        sys.exit(1)
    run(factory, n)

import asyncio
import websockets
import json
import neat
import os
import ssl
import math
import pickle
import sys
import random
from neat.math_util import mean

# --- CONFIGURATION ---
SERVER_URL = "wss://re-genes.is/ws/join?species=NEAT_Evo"


def _make_ssl_context():
    """Contexto TLS robusto para conectar em wss://re-genes.is.

    Causa raiz do 'certificate has expired': o trust store do WINDOWS desta maquina
    tem um root vencido do Let's Encrypt (DST Root CA X3, expirou em 2021) e o OpenSSL
    do Python monta o caminho por ele em vez do ISRG atual. A cadeia do servidor esta
    valida — o problema e o store local. Solucao: usar o bundle atualizado do certifi
    em vez do Windows store.

    Tambem removemos VERIFY_X509_STRICT (Python 3.13+ liga por padrao): se o Avast
    estiver interceptando TLS, o root dele tem 'Basic Constraints' nao-critical e o
    modo estrito reprova. Root extra (ex.: o do Avast) pode ser injetado via env var
    REGENES_CA_EXTRA=caminho\\para\\ca.pem.

    Escape opt-in: REGENES_INSECURE_TLS=1 desliga a verificacao (trafego aqui e so
    telemetria de jogo, sem segredo). Fix permanente ideal: manter o Windows/Root
    store atualizado (Windows Update) ou excluir o host do scan HTTPS do Avast.
    """
    if os.getenv("REGENES_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("⚠️  REGENES_INSECURE_TLS=1 -> verificacao de certificado DESLIGADA")
        return ctx
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    extra = os.getenv("REGENES_CA_EXTRA")
    if extra and os.path.exists(extra):
        try:
            ctx.load_verify_locations(extra)
        except Exception:
            pass
    return ctx


# Reusado por todas as conexoes (so relevante para wss://).
SSL_CONTEXT = _make_ssl_context() if SERVER_URL.startswith("wss") else None
CONFIG_FILE = "config-feedforward"
CHECKPOINT_PREFIX = "neat-checkpoint-v2-"  # v2 = predação (104 in / 9 out). v1 preservado.
AUTOSAVE_INTERVAL = 300  # Saves every 5 minutes (approx) or by event

import itertools
import gzip

class PicklableCount:
    """A replacement for itertools.count that CAN be pickled."""
    def __init__(self, start=0):
        self._current = start
        
    def __next__(self):
        val = self._current
        self._current += 1
        return val
        
    def __iter__(self):
        return self
        
    def get_current(self):
        return self._current

class ContinuousPopulation:
    """
    Manages a NEAT population for continuous (steady-state) evolution.
    Instead of generations, we maintain a pool of genomes.
    When one dies, we immediately breed a replacement.
    """
    def __init__(self, config_path, checkpoint_file=None):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        
        loaded = False
        # Só tenta carregar se o arquivo existe E não está vazio (0 bytes = escrita
        # interrompida). Antes, um checkpoint vazio derrubava o boot ('Ran out of input').
        if checkpoint_file and os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
            print(f"📂 Loading checkpoint: {checkpoint_file}")
            try:
                with gzip.open(checkpoint_file) as f:
                    self.p = pickle.load(f)
                if hasattr(self.p, 'config'):
                    self.config = self.p.config
                else:
                    self.p.config = self.config
                loaded = True
            except Exception as e:
                print(f"⚠️ Falha no load custom ({e}); tentando restore legado…")
                try:
                    self.p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
                    self.config = self.p.config
                    loaded = True
                except Exception as e2:
                    print(f"⚠️ Restore legado também falhou ({e2}). Começando do zero.")
        if not loaded:
            print("🌱 Creating new population...")
            self.p = neat.Population(self.config)
            
        # Initialize internal NEAT structures if new
        if not hasattr(self.p, 'generation'):
            self.p.generation = 0
        if not hasattr(self.p, 'species'):
            self.p.species = neat.DefaultSpeciesSet(self.config.species_set_config, self.p.reporters)
            
        # FIX: itertools.count cannot be pickled. Replace with custom one.
        # Check both population.reproduction and config.genome_config
        repro = self.p.reproduction
        if hasattr(repro, 'genome_indexer'):
            next_id = 0
            if self.p.population:
                next_id = max(self.p.population.keys()) + 1
            repro.genome_indexer = PicklableCount(next_id)

        # FIX: The node indexer must be synchronized across the entire population's genome history
        g_config = self.config.genome_config
        if hasattr(g_config, 'node_indexer'):
            max_node_id = -1
            for g in self.p.population.values():
                if g.nodes:
                    max_node_id = max(max_node_id, max(g.nodes.keys()))
            
            # Ensure the indexer starts past the highest known node ID
            g_config.node_indexer = PicklableCount(max_node_id + 1)

        # FIX: The node indexer might also be an itertools.count and needs to be synchronized.
        # It's usually found in the genome_config or as part of the reproduction object.
        if hasattr(self.config.genome_config, 'node_indexer'):
            if isinstance(self.config.genome_config.node_indexer, itertools.count):
                # Find the maximum node ID currently in use across the entire population
                max_node_id = -1
                for g in self.p.population.values():
                    if g.nodes:
                        max_node_id = max(max_node_id, max(g.nodes.keys()))
                
                # Input nodes have negative IDs in neat-python, we only care about new node IDs (which are positive)
                # but max() will handle it.
                self.config.genome_config.node_indexer = PicklableCount(max_node_id + 1)
            
        # We need to manually handle speciation and reproduction
        # Ensure initial speciation
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        
        # Keep track of active genomes (being simulated) to avoid simulating same genome twice
        self.active_genome_ids = set()
        self.deaths_since_gen_inc = 0

    def get_genome(self):
        """
        Returns a genome to simulate.
        Strategy:
        1. If there are unsued genomes in current pool, return one.
        2. If all are busy/used, generate a NEW one (offspring).
        """
        # Try to find an inactive genome from current pool
        for gid, genome in self.p.population.items():
            if gid not in self.active_genome_ids and getattr(genome, 'fitness', None) is None:
                self.active_genome_ids.add(gid)
                return gid, genome
                
        # If we are here, all current genomes are either active or have fitness.
        # It's time to breed a replacement for the "worst" or just adds to pool?
        # Steady State: Remove worst, Breed new.
        # Or simply Breed new and grow pool? NEAT usually has fixed pop size.
        
        # Simple Steady State Logic:
        # 1. Speciate (Update species with current fitnesses)
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        
        # 2. Spawn Child
        # We use DefaultReproduction's logic but manually. 
        # Since accessing internal reproduction logic is hard, we implement basic tournament here.
        
        child = self._breed_child()
        

        # 3. Add to population
        if len(self.p.population) > self.config.pop_size * 2: # Cull looser cap
             self._cull_population()
             
        self.p.population[child.key] = child
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        
        self.active_genome_ids.add(child.key)
        return child.key, child

    def report_death(self, genome_id, fitness):
        """Called when ameba dies."""
        if genome_id in self.p.population:
            self.p.population[genome_id].fitness = fitness
        
        if genome_id in self.active_genome_ids:
            self.active_genome_ids.remove(genome_id)
            
        # Increment generation count every N deaths to trigger stagnation logic
        self.deaths_since_gen_inc += 1
        if self.deaths_since_gen_inc >= self.config.pop_size:
            self.p.generation += 1
            self.deaths_since_gen_inc = 0
            # Speciate occasionally
            self.p.species.speciate(self.config, self.p.population, self.p.generation)
            print(f"📈 [GEN] Incrementing generation to {self.p.generation} | Pop: {len(self.p.population)}")

    def save_checkpoint(self):
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        filename = os.path.join(os.path.dirname(__file__), f"{CHECKPOINT_PREFIX}auto")
        # Escrita ATÔMICA: grava num temp e renomeia. Se interromper no meio, o
        # checkpoint bom antigo permanece intacto (nunca gera arquivo de 0 bytes).
        tmp = filename + ".tmp"
        with gzip.open(tmp, 'wb', compresslevel=5) as f:
            pickle.dump(self.p, f)
        os.replace(tmp, filename)

    def _breed_child(self):
        valid_genomes = [g for g in self.p.population.values() if g.fitness is not None]
        if not valid_genomes:
            # Setup innovation tracker if missing (Critical for first run/new pop)
            tracker = getattr(self.config.genome_config, 'innovation_tracker', None)
            if tracker is None:
                if hasattr(self.p.reproduction, 'innovation_tracker'):
                    self.config.genome_config.innovation_tracker = self.p.reproduction.innovation_tracker
            
            gid = self.p.reproduction.ancestry.next()
            child = self.config.genome_type(gid)
            child.configure_new(self.config.genome_config)
            return child
        
        # Tournament Selection - Favor active/fit parents
        parent1 = self._tournament(valid_genomes)
        parent2 = self._tournament(valid_genomes)
        
        gid = next(self.p.reproduction.genome_indexer)
        child = self.config.genome_type(gid)
        
        # Setup innovation tracker for mutation
        # Check if missing OR None
        tracker = getattr(self.config.genome_config, 'innovation_tracker', None)
        if tracker is None:
             if hasattr(self.p.reproduction, 'innovation_tracker'):
                 self.config.genome_config.innovation_tracker = self.p.reproduction.innovation_tracker
             else:
                 # Fallback: Create new tracker if none exists
                 print("⚠️ Creating new InnovationTracker (Fallback)")
                 from neat.innovation import InnovationTracker
                 self.config.genome_config.innovation_tracker = InnovationTracker()
                 # Attach to reproduction to persist it
                 self.p.reproduction.innovation_tracker = self.config.genome_config.innovation_tracker

        child.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # CRITICAL SAFETY: Ensure node_indexer is ahead of any node IDs inherited from parents
        for node_id in child.nodes.keys():
            while node_id >= self.config.genome_config.node_indexer.get_current():
                next(self.config.genome_config.node_indexer)
        
        child.mutate(self.config.genome_config)
        return child

    def _tournament(self, genomes, k=3): # Increased K for stronger selection pressure
        candidates = random.sample(genomes, min(len(genomes), k))
        return max(candidates, key=lambda g: g.fitness)

    def _cull_population(self):
        sorted_pop = sorted(self.p.population.items(), key=lambda item: item[1].fitness if item[1].fitness is not None else -9999)
        to_remove_count = int(len(sorted_pop) * 0.2) # Cull 20%
        for i in range(to_remove_count):
            gid, _ = sorted_pop[i]
            if gid not in self.active_genome_ids:
                del self.p.population[gid]

# Global Stats
ACTION_STATS = {"UP":0, "DOWN":0, "LEFT":0, "RIGHT":0, "STAY":0}
TOTAL_ACTIONS = 0
SESSION_TOTAL = 0

class NeatAmeba:
    def __init__(self, genome, config, genome_id, population_manager):
        self.genome = genome
        self.config = config
        self.genome_id = genome_id
        self.manager = population_manager
        
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.fitness = 0.0
        self.alive = True
        self.max_ticks = 2000 
        self.energy_gained = 0
        self.last_energy = 100
        self.visited_cells = set()
        self.food_eaten_count = 0
        self.start_pos = (0, 0)
        self.current_pos = (0, 0)
        
        # Endorphin System (0-100)
        self.endorphin = 50.0  # Starts neutral
        self.accumulated_endorphin = 0.0 # For fitness calculation
        
        # 🎥 Replay History
        self.history = []

    def _save_replay(self, fitness, ticks):
        filename = f"replays/replay_G{self.genome_id}_F{int(fitness)}.json.gz"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        data = {
            "genome_id": self.genome_id,
            "fitness": fitness,
            "total_ticks": ticks,
            "history": self.history
        }
        
        try:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            print(f"🎬 [REPLAY] Saved {filename} ({len(self.history)} frames)")
        except Exception as e:
            print(f"❌ Failed to save replay: {e}")

    async def run(self):
        global TOTAL_ACTIONS, SESSION_TOTAL
        try:
            async with websockets.connect(SERVER_URL, ssl=SSL_CONTEXT) as websocket:
                # Handshake
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                self.my_id = welcome_data.get("id")
                
                # Capture Genome Stats for Dynamic Normalization
                stats = welcome_data.get("stats", {})
                self.stomach_size = stats.get("stomach_size", 200.0)
                self.digestion_rate = stats.get("digestion_rate", 1.0)
                
                self.start_pos = (welcome_data.get('x', 0), welcome_data.get('y', 0))
                self.current_pos = self.start_pos
                
                tick_count = 0
                while self.alive and tick_count < self.max_ticks:
                    try:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        
                        if data['type'] == 'UPDATE':
                            if not data['alive']:
                                self.alive = False
                            
                            current_energy = data.get('energy', 0)
                            delta = current_energy - self.last_energy
                            if delta > 0:
                                self.energy_gained += delta
                                self.food_eaten_count += 1
                                # 💊 Endorphin: Eating is joy!
                                # REHAB: Massive reward to make eating the primary goal
                                self.endorphin += 100.0
                                
                                if self.food_eaten_count % 5 == 1:
                                    print(f"🍏 [G{self.genome_id}] Ate food! Total: {self.food_eaten_count} | Endorphin: {self.endorphin:.1f}")
                            
                            self.last_energy = current_energy
                            
                            # Track exploration
                            pos = (data.get('x'), data.get('y'))
                            if pos[0] is not None and pos != self.current_pos:
                                # 💊 Endorphin: Movement reward (Make it sustainable!)
                                # REHAB: Reduced from 0.5 to 0.3 to discourage aimless running
                                self.endorphin += 0.3
                                
                                if pos not in self.visited_cells:
                                    self.visited_cells.add(pos)
                                    # 💊 Endorphin: Discovery reward!
                                    self.endorphin += 5.0
                                
                                self.current_pos = pos
                            
                            continue
    
                        if data['type'] == 'TICK':
                            vision = data.get('vision') 
                            energy = data.get('energy', 0)
                            stomach = data.get('stomach', 0)
                            
                            # 💊 Endorphin: Natural Decay (Boredom/Stress)
                            # Reduced to 0.2 so moving (+0.3) is net positive (+0.1)
                            self.endorphin -= 0.2
                            
                            # 💊 Endorphin: Starvation Stress
                            # REHAB: Panic if energy drops below 50% (was 20%)
                            if energy < (self.stomach_size * 0.5):
                                self.endorphin -= 2.0
                            
                            # Clamp Endorphin (0-100)
                            self.endorphin = max(0.0, min(100.0, self.endorphin))
                            
                            # Integrate happiness over time (Area Under Curve)
                            self.accumulated_endorphin += self.endorphin
                            
                            inputs = self.process_inputs(vision, energy, stomach)
                            outputs = self.net.activate(inputs)
                            
                            action_idx = outputs.index(max(outputs))
                            
                            cmd = "stay"
                            direction = "UP"
                            act_label = "STAY"
                            
                            if action_idx == 0: 
                                cmd, direction, act_label = "move", "UP", "UP"
                            elif action_idx == 1: 
                                cmd, direction, act_label = "move", "DOWN", "DOWN"
                            elif action_idx == 2: 
                                cmd, direction, act_label = "move", "LEFT", "LEFT"
                            elif action_idx == 3: 
                                cmd, direction, act_label = "move", "RIGHT", "RIGHT"
                            
                            # Stats Update
                            ACTION_STATS[act_label] += 1
                            TOTAL_ACTIONS += 1
                            SESSION_TOTAL += 1
                            
                            # 🎥 Replay Recording (In-Memory)
                            # We store what she SAW and what she DID.
                            step_record = {
                                "t": tick_count,
                                "x": self.current_pos[0],
                                "y": self.current_pos[1],
                                "e": round(energy, 1),
                                "h": round(self.endorphin, 1), # Happiness
                                "a": cmd,
                                "d": direction,
                                "v": vision # The subjective reality (3x9x9)
                            }
                            self.history.append(step_record)
                            
                            await websocket.send(json.dumps({
                                "action": cmd,
                                "direction": direction
                            }))

                            if self.genome_id % 10 == 0 and tick_count % 20 == 0:
                               self._log_debug(inputs, outputs, act_label)
                            
                            tick_count += 1
                            
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"⚠️ [G{self.genome_id}] Error: {e}")
                        break
                        
                # 🏆 Fitness V7: Survival + Foraging (gradiente forte e selecionável)
                # A V6 ("felicidade média") era dominada por +0.3/movimento e +5/célula nova,
                # então vagar sem rumo pontuava quase igual a comer. Resultado: 7616 vidas com
                # fitness PLANO (~31) e zero aprendizado. Aqui o sinal vem de comer e sobreviver:
                #   - cada comida ingerida vale MUITO (evento raro e o real objetivo)
                #   - energia total absorvida dá um sinal denso (recompensa parcial por caçar)
                #   - sobreviver mais ticks é bônus secundário
                #   - exploração fica como termo minúsculo, só para bootstrap inicial
                FOOD_REWARD = 200.0      # por bocado de comida efetivamente comido
                ENERGY_REWARD = 1.0      # por unidade de energia absorvida
                SURVIVAL_REWARD = 1.0    # por tick vivo
                EXPLORE_REWARD = 0.25    # por célula nova visitada (bootstrap)

                final_fitness = (
                    self.food_eaten_count * FOOD_REWARD
                    + self.energy_gained * ENERGY_REWARD
                    + tick_count * SURVIVAL_REWARD
                    + len(self.visited_cells) * EXPLORE_REWARD
                )

                # Log performance
                self._save_to_performance_log(final_fitness, tick_count, self.food_eaten_count)

                # Hall of Fame / Replay: limiares recalibrados para a escala V7.
                # (comer ~10 alimentos + sobreviver ~500 ticks ≈ 2500+)
                if final_fitness > 1500:
                    self._update_hall_of_fame(final_fitness, tick_count)
                if final_fitness > 2500:
                    # 🎥 Save Replay for Geniuses
                    self._save_replay(final_fitness, tick_count)

                self.manager.report_death(self.genome_id, final_fitness)

        except Exception as e:
            print(f"❌ [G{self.genome_id}] Setup Error: {e}")
            self.manager.report_death(self.genome_id, 0)


    def _log_debug(self, inputs, outputs, dir):
        def fmt(x): return f"{x:.1f}"
        # Inputs: 0=Bias, 1=Energy, 2=Stomach, 3=Endorphin
        print(f"[G{self.genome_id}] E:{fmt(inputs[1])} S:{fmt(inputs[2])} ❤️:{fmt(inputs[3])} -> {dir}")

    def process_inputs(self, vision, energy, stomach):
        """
        New Topology: 79 Inputs (5x5 Vision + Status + Endorphin)
        I0: Bias
        I1: Energy (norm)
        I2: Stomach (norm)
        I3: Endorphin (norm)
        I4-I28: Walls (25 Cells)
        I29-I53: Scent (25 Cells)
        I54-I78: Enemies (25 Cells)
        """
        if not vision or len(vision[0]) < 9: 
            return [0.0] * 79
        
        # Center is at 4,4 (Radius 4 in 9x9 grid)
        cx, cy = 4, 4
        
        # 5x5 Grid (including center 4,4)
        inputs = [1.0] # Bias
        inputs.append(min(energy, self.stomach_size) / self.stomach_size)
        inputs.append(min(stomach, self.stomach_size) / self.stomach_size)
        
        # New Input: Endorphin Level (Normalized 0.0 - 1.0)
        inputs.append(self.endorphin / 100.0)
        
        # We collect all 25 cells in range of -2 to +2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = cy + dy, cx + dx
                # Wall
                inputs.append(1.0 if vision[0][y][x] > 0 else 0.0)
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = cy + dy, cx + dx
                # Scent
                inputs.append(vision[1][y][x])
                
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = cy + dy, cx + dx
                # Enemies
                inputs.append(vision[2][y][x])
            
        return inputs

    def _save_to_performance_log(self, fitness, ticks, food):
        file = os.path.join(os.path.dirname(__file__), "neat_performance.csv")
        exists = os.path.exists(file)
        with open(file, "a", encoding='utf-8') as f:
            if not exists:
                f.write("id,fitness,ticks,food_eaten\n")
            f.write(f"{self.genome_id},{fitness:.2f},{ticks},{food}\n")

    def _update_hall_of_fame(self, fitness, ticks):
        file = os.path.join(os.path.dirname(__file__), "NEAT_HALL_OF_FAME.md")
        with open(file, "a", encoding='utf-8') as f:
            f.write(f"| G{self.genome_id} | {fitness:.2f} | {ticks} | {self.food_eaten_count} |\n")

async def run_simulation(target_count):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, CONFIG_FILE)
    
    # Init Continuous Population Manager
    pop = ContinuousPopulation(config_path, checkpoint_file=f"{CHECKPOINT_PREFIX}auto")
    
    print(f"🚀 Starting Continuous NEAT | Target: {target_count} Amebas")
    
    active_tasks = set()
    
    # Autosaving Loop
    async def auto_saver():
        while True:
            await asyncio.sleep(AUTOSAVE_INTERVAL)
            pop.save_checkpoint()

    # Stats Reporter Loop
    async def stats_reporter():
        global TOTAL_ACTIONS, ACTION_STATS
        while True:
            await asyncio.sleep(10)
            if TOTAL_ACTIONS > 0:
                print(f"\n📊 [STATS] Windowed Distribution (Last 10s):")
                total = TOTAL_ACTIONS
                for k in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]:
                    v = ACTION_STATS[k]
                    pct = (v / total) * 100
                    bar = "█" * int(pct/5)
                    print(f"   {k:5}: {pct:5.1f}% {bar}")
                print(f"   Window Actions: {total} | Session Total: {SESSION_TOTAL}\n")
                
                # Reset Window Stats
                TOTAL_ACTIONS = 0
                for k in ACTION_STATS: ACTION_STATS[k] = 0
            
    asyncio.create_task(auto_saver())
    asyncio.create_task(stats_reporter())

    # Main Spawn Loop
    try:
        while True:
            # Refill
            while len(active_tasks) < target_count:
                # 1. Get Genome
                gid, genome = pop.get_genome()
                
                # 2. Create Agent
                ameba = NeatAmeba(genome, pop.config, gid, pop)
                
                # 3. Spawn Task
                task = asyncio.create_task(ameba.run())
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)
                
                # Stagger spawns slightly to avoid connection bursts
                await asyncio.sleep(0.1)
                
            # Wait for something to finish
            if active_tasks:
                await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("\n🛑 Simulation Cancelled.")
    finally:
        print("\n💾 Saving checkpoint before exit (Finally)...")
        pop.save_checkpoint()

if __name__ == '__main__':
    # Default de população simultânea. Neuroevolução precisa de vários indivíduos
    # vivos ao mesmo tempo (throughput + competição). Rode `python client_neat.py 20`
    # para escalar; respeite o limite de CPU do servidor (docker: 0.5 CPU).
    target = 8
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except:
            pass
    
    try:
        # Windows specifics for better signal handling
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        asyncio.run(run_simulation(target))
    except KeyboardInterrupt:
        print("\n👋 Stopping (KeyboardInterrupt)...")
        # Ensure checkpoint is saved even on hard interrupt if `finally` block didn't catch it
        # Note: In robust systems, we'd rely on the `finally` block in `run_simulation`, 
        # but asyncio.run can sometimes swallow the context.
        # However, `pop` is local to `run_simulation`, so we can't access it here easily.
        # The `finally` block inside `run_simulation` IS the correct place. 
        # The issue might be that `gzip` flush didn't complete.
        pass

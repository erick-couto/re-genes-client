import asyncio
import websockets
import json
import neat
import os
import math
import pickle
import sys
import random
from neat.math_util import mean

# --- CONFIGURAÇÃO ---
SERVER_URL = "wss://re-genes.is/ws/join?species=NEAT_Evo"
CONFIG_FILE = "config-feedforward"
CHECKPOINT_PREFIX = "neat-checkpoint-continuous-"
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
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"📂 Loading checkpoint: {checkpoint_file}")
            try:
                with gzip.open(checkpoint_file) as f:
                    self.p = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load custom checkpoint: {e}")
                print("🔄 Attempting legacy restore...")
                self.p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        else:
            print("🌱 Creating new population...")
            self.p = neat.Population(self.config)
            
        # Initialize internal NEAT structures if new
        if not hasattr(self.p, 'generation'):
            self.p.generation = 0
        if not hasattr(self.p, 'species'):
            self.p.species = neat.DefaultSpeciesSet(self.config.species_set_config, self.p.reporters)
            
        # FIX: itertools.count cannot be pickled. Replace with custom one if needed.
        if hasattr(self.p.reproduction, 'genome_indexer'):
            if isinstance(self.p.reproduction.genome_indexer, itertools.count):
                # We need to guess current ID. DefaultReproduction doesn't expose it easily unless we iterate?
                # Actually, max(genome keys) + 1 is safe.
                next_id = 0
                if self.p.population:
                    next_id = max(self.p.population.keys()) + 1
                self.p.reproduction.genome_indexer = PicklableCount(next_id)
            
        # We need to manually handle speciation and reproduction
        # Ensure initial speciation
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        
        # Keep track of active genomes (being simulated) to avoid simulating same genome twice
        self.active_genome_ids = set()

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

    def save_checkpoint(self):
        self.p.species.speciate(self.config, self.p.population, self.p.generation)
        filename = f"{CHECKPOINT_PREFIX}auto"
        # print(f"💾 Saving checkpoint to {filename}...")
        with gzip.open(filename, 'w', compresslevel=5) as f:
            pickle.dump(self.p, f)

    def _breed_child(self):
        valid_genomes = [g for g in self.p.population.values() if g.fitness is not None]
        if not valid_genomes:
            # Setup innovation tracker if missing (Critical for first run/new pop)
            tracker = getattr(self.config.genome_config, 'innovation_tracker', None)
            if tracker is None:
                if hasattr(self.p.reproduction, 'innovation_tracker'):
                    self.config.genome_config.innovation_tracker = self.p.reproduction.innovation_tracker
            
            return self.config.genome_type(self.p.reproduction.ancestry.next())
        
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
        
        # Bio-Realism: Dynamic normalization based on actual genome stats
        self.stomach_size = 200.0 # Default fallback
        self.digestion_rate = 1.0 # Default fallback
        
    async def run(self):
        global TOTAL_ACTIONS
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                # Handshake
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                self.my_id = welcome_data.get("id")
                
                # Capture Genome Stats for Dynamic Normalization
                stats = welcome_data.get("stats", {})
                self.stomach_size = stats.get("stomach_size", 200.0)
                self.digestion_rate = stats.get("digestion_rate", 1.0)
                
                # print(f"🧬 [G{self.genome_id}] Spawned as {self.my_id} | Cap: {self.stomach_size}")
                
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
                            self.last_energy = current_energy
                            continue
    
                        if data['type'] == 'TICK':
                            vision = data.get('vision') 
                            energy = data.get('energy', 0)
                            stomach = data.get('stomach', 0)
                            
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
                            
                            await websocket.send(json.dumps({
                                "action": cmd,
                                "direction": direction
                            }))
                            
                            if self.genome_id % 5 == 0 and tick_count % 50 == 0:
                               # Log occasionally for sample
                               pass 
                            
                            tick_count += 1
                            
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"⚠️ [G{self.genome_id}] Error: {e}")
                        break
                        
                # End of Life
                # Fitness strategy:
                # 1. Heavily Reward Eating (Energy Gained)
                # 2. Lightly Reward Survival (Tick Count) but capped or scaled down
                # If they just stay still, they get ~200 ticks. If they eat 1 food (10 energy), they should beat that.
                
                eating_score = self.energy_gained * 20.0 # 1 Food = 200 points
                survival_score = tick_count * 0.1       # 2000 ticks = 200 points
                
                final_fitness = eating_score + survival_score
                
                # print(f"💀 [G{self.genome_id}] Died. Fitness: {final_fitness:.1f} (Eat: {self.energy_gained})")
                self.manager.report_death(self.genome_id, final_fitness)

        except Exception as e:
            print(f"❌ [G{self.genome_id}] Setup Error: {e}")
            self.manager.report_death(self.genome_id, 0)


    def _log_debug(self, inputs, outputs, dir):
        def fmt(x): return f"{x:.1f}"
        print(f"[G{self.genome_id}] E:{fmt(inputs[1])} S:{fmt(inputs[2])} -> {dir}")

    def process_inputs(self, vision, energy, stomach):
        """
        New Topology: 27 Inputs
        I0: Bias
        I1: Energy (norm)
        I2: Stomach (norm)
        I3-I10: Walls (8 Neighbors)
        I11-I18: Food (8 Neighbors - Gradients)
        I19-I26: Enemies (8 Neighbors - Size Gene)
        """
        if not vision or len(vision[0]) < 9: 
            return [0.0] * 27
        
        # Center is at 4,4 (Radius 4)
        cx, cy = 4, 4
        
        # Directions for 8 neighbors (Moore): N, S, W, E, NW, NE, SW, SE
        # Note: Array is [y][x]. 
        # Up (N) is y-1. Down (S) is y+1.
        neighbors = [
            (cy-1, cx),   # N
            (cy+1, cx),   # S
            (cy, cx-1),   # W
            (cy, cx+1),   # E
            (cy-1, cx-1), # NW
            (cy-1, cx+1), # NE
            (cy+1, cx-1), # SW
            (cy+1, cx+1)  # SE
        ]
        
        # Extract Layers
        layer_walls = vision[0]
        layer_scent = vision[1]
        layer_enemy = vision[2]
        
        inputs = []
        
        # 1. Body Stats
        inputs.append(1.0) # Bias
        # Bio-Realism V2: Dynamic Normalization
        inputs.append(min(energy, self.stomach_size) / self.stomach_size)
        inputs.append(min(stomach, self.stomach_size) / self.stomach_size)
        
        # 2. Walls (8)
        for y, x in neighbors:
            inputs.append(1.0 if layer_walls[y][x] > 0 else 0.0)
            
        # 3. Food (8) - Direct Decimal Gradient
        for y, x in neighbors:
            inputs.append(layer_scent[y][x])
            
        # 4. Enemies (8) - Size Gene
        for y, x in neighbors:
            inputs.append(layer_enemy[y][x])
            
        return inputs

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
        while True:
            await asyncio.sleep(10)
            if TOTAL_ACTIONS > 0:
                print("\n📊 [STATS] Action Distribution (Last 10s):")
                total = TOTAL_ACTIONS
                for k, v in ACTION_STATS.items():
                    pct = (v / total) * 100
                    bar = "█" * int(pct/5)
                    print(f"   {k:5}: {pct:5.1f}% {bar}")
                print(f"   Total Actions: {total}\n")
            
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
    finally:
        print("\n💾 Saving checkpoint before exit...")
        pop.save_checkpoint()

if __name__ == '__main__':
    target = 1
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except:
            pass
    
    try:
        asyncio.run(run_simulation(target))
    except KeyboardInterrupt:
        print("\n👋 Stopping...")

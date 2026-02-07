import asyncio
import websockets
import json
import random
import os
import sys
from collections import  defaultdict

# --- CONFIGURAÇÃO ---
SERVER_URL = "wss://re-genes.is/ws/join?species=Memoriam"
Q_TABLE_BASE_NAME = "qtable_memoriam"
Q_TABLE_BASE_NAME = "qtable_memoriam"
DEFAULT_BATCH_SIZE = 1  # Standard Default if no args
TICKS_PER_GEN = 2000 # Tempo máximo de vida por geração

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.01  # Lower floor for better long-term precision
EPSILON_DECAY = 0.99 # Smoother decay per generation (was 0.90)

class MemoriamBrain:
    """
    Represents the brain (Q-Table) of a specific SPECIES/PHENOTYPE.
    Multiple amebas can share this instance if they are of the same species.
    """
    def __init__(self, species_name):
        self.species_name = species_name
        self.q_table = {}
        self.epsilon = EPSILON_START
        self.filename = f"{Q_TABLE_BASE_NAME}_{species_name}.json"
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.q_table = data.get("q_table", {})
                    self.epsilon = data.get("epsilon", EPSILON_START)
                print(f"📖 [{self.species_name}] Memory loaded: {len(self.q_table)} states. Eps: {self.epsilon:.2f}")
            except Exception as e:
                print(f"⚠️ Error loading {self.filename}: {e}")
                self.q_table = {}
        else:
            print(f"✨ [{self.species_name}] New species discovered. Empty brain.")

    def save(self):
        try:
            temp_file = self.filename + ".tmp"
            
            # Round values for cleaner JSON
            clean_table = {}
            for state, actions in self.q_table.items():
                clean_table[state] = {k: round(v, 4) for k, v in actions.items()}

            with open(temp_file, 'w') as f:
                json.dump({
                    "epsilon": self.epsilon,
                    "q_table": clean_table
                }, f)
            
            if os.path.exists(self.filename):
                os.remove(self.filename)
            os.rename(temp_file, self.filename)
            print(f"💾 [{self.species_name}] Saved. {len(self.q_table)} states. Eps: {self.epsilon:.2f}")
        except Exception as e:
            print(f"❌ Error saving {self.filename}: {e}")

    def get_action(self, state_key):
        actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        if state_key not in self.q_table:
            return random.choice(actions)
            
        q_values = self.q_table[state_key]
        return max(q_values, key=q_values.get)

    def update(self, state, action, reward, next_state):
        # Init states if new
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]}

        old_val = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        
        new_val = old_val + ALPHA * (reward + GAMMA * next_max - old_val)
        self.q_table[state][action] = new_val

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


class BrainManager:
    """Manages multiple brains based on Phenotype"""
    def __init__(self):
        self.brains = {} # Dict[species_name, MemoriamBrain]

    def get_brain(self, species_desc):
        # Sanitize name (Ex: 'Giant Slow (Offspring of..)' -> 'Giant_Slow')
        clean_name = species_desc.split("(")[0].strip().replace(" ", "_")
        if not clean_name: clean_name = "Unknown"
        
        if clean_name not in self.brains:
            self.brains[clean_name] = MemoriamBrain(clean_name)
        return self.brains[clean_name]

    def save_all(self):
        for brain in self.brains.values():
            brain.save()


# --- LÓGICA DA AMEBA INDIVIDUAL ---
class AgentAmeba:
    def __init__(self, ameba_id, manager):
        self.id = ameba_id
        self.manager = manager
        self.brain = None
        self.last_state = None
        self.last_action = None
        self.last_energy = 100
        self.alive = True
    
    def process_vision(self, vision, energy):
        """Transforms vision into State (String Key)"""
        if not vision: return "BLIND"
        
        # Center Crop 3x3 (Channel 0=Wall, 1=Scent)
        # Vision Radius 4 (9x9 grid), Center at 4,4
        parts = []
        cy, cx = 4, 4
        
        for y in range(cy-1, cy+2):
            for x in range(cx-1, cx+2):
                wall = 1 if vision[0][y][x] > 0 else 0
                
                # Scent is now rounded by server, but we quantize for smaller state space
                scent = vision[1][y][x]
                s_lvl = 0
                if scent > 0.5: s_lvl = 2
                elif scent > 0.05: s_lvl = 1
                
                parts.append(f"{wall}{s_lvl}")
                
        e_state = "OK"
        if energy < 20: e_state = "CRIT"
        elif energy < 50: e_state = "LOW"
        
        return "".join(parts) + "_" + e_state

    async def run(self):
        try:
            async with websockets.connect(SERVER_URL) as ws:
                # 1. Handshake
                welcome = json.loads(await ws.recv())
                server_id = welcome['id']
                species = welcome['species']
                
                # 2. Get Shared Brain
                self.brain = self.manager.get_brain(species)
                print(f"✨ {server_id} connected. Species: {self.brain.species_name}")

                # 3. Game Loop
                tick_count = 0
                while self.alive and tick_count < TICKS_PER_GEN:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    if data['type'] == 'UPDATE':
                        self.alive = data['alive']
                        current_energy = data.get('energy', 0)
                        
                        if not self.alive:
                            # Death penalty
                            if self.last_state and self.last_action:
                                self.brain.update(self.last_state, self.last_action, -100, "DEATH")
                        continue

                    if data['type'] == 'TICK':
                        vision = data.get('vision')
                        energy = data.get('energy', self.last_energy)
                        
                        current_state = self.process_vision(vision, energy)
                        
                        # Learn from previous step
                        if self.last_state and self.last_action:
                            reward = 0
                            delta = energy - self.last_energy
                            if delta > 0: reward = 50 
                            elif delta == 0: reward = -0.1
                            else: reward = -1
                            
                            self.brain.update(self.last_state, self.last_action, reward, current_state)

                        # Act
                        action = self.brain.get_action(current_state)
                        
                        direction = "UP"
                        cmd = "move"
                        if action == "STAY":
                            cmd = "stay"
                        else:
                            direction = action
                            
                        await ws.send(json.dumps({"action": cmd, "direction": direction}))
                        
                        # Update Memory
                        self.last_state = current_state
                        self.last_action = action
                        self.last_energy = energy
                        tick_count += 1
                        
        except Exception as e:
            print(f"⚠️ Agent Error: {e}")

# --- CONTROLE DA GERAÇÃO ---
async def periodic_save(manager):
    """Saves memory every 30 seconds without stopping simulation"""
    while True:
        await asyncio.sleep(30)
        print("\n💾 Periodic Auto-Save...")
        manager.save_all()
        
        # Decay Epsilon Periodically
        for brain in manager.brains.values():
            brain.decay_epsilon()

async def main():
    manager = BrainManager()
    
    # Init Batch from Arg
    target_population = DEFAULT_BATCH_SIZE
    if len(sys.argv) > 1:
        try:
            target_population = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ Invalid argument: {sys.argv[1]}. Using Default: {target_population}")
            
    print(f"🚀 Starting Memoriam Simulator | Target: {target_population} Simultaneous Amebas")

    # Start Auto-Saver
    asyncio.create_task(periodic_save(manager))
    
    active_tasks = set()
    ameba_counter = 0

    while True:
        # 1. Fill Population
        while len(active_tasks) < target_population:
            ameba_counter += 1
            ameba = AgentAmeba(ameba_counter, manager)
            # Create task and add to set
            task = asyncio.create_task(ameba.run())
            active_tasks.add(task)
            
            # Remove reference when done
            task.add_done_callback(active_tasks.discard)
            
        # 2. Wait for ANY death (to replenish immediately)
        if active_tasks:
            # Wait for at least one task to finish
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Optional: Log death count or replacements
            # print(f"💀 {len(done)} ameba(s) morreram. Reabastecendo...")
        else:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Encerrando...")

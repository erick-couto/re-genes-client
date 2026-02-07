import neat
import os
import glob
import gzip
import pickle
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import PicklableCount so it can be unpickled
class PicklableCount:
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

# Config
CONFIG_FILE = "config-feedforward"

def load_latest_checkpoint():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(local_dir, 'neat-checkpoint-continuous-*')
    list_of_files = glob.glob(pattern) 
    if not list_of_files:
        print("FAILED: No checkpoints found.")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_file}")
    
    try:
        with gzip.open(latest_file, 'rb') as f:
            return pickle.load(f)
    except:
        return neat.Checkpointer.restore_checkpoint(latest_file)

def get_best_genome(p):
    best = None
    best_fitness = -99999.0
    for g_id, genome in p.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best = genome
    return best

def simulate_brain(net, best_genome):
    print("\n--- BRAIN SIMULATION (75 Inputs) ---")
    
    # Scenarios: 75 inputs
    # Bias (1) + En (1) + St (1) + Walls (24) + Scent (24) + Enemies (24)
    def make_inputs(en=1.0, st=0.0, walls=None, scents=None):
        base = [1.0, en, st]
        w = [0.0] * 24
        s = [0.0] * 24
        e = [0.0] * 24
        if walls:
            for idx in walls: w[idx] = 1.0
        if scents:
            for idx in scents: s[idx] = 1.0
        return base + w + s + e

    scenarios = [
        ("Empty (Full Energy)", make_inputs(en=1.0, st=0.0)),
        ("Wall UP",            make_inputs(en=0.8, walls=[7])), # 5x5 grid, index 7 is (1,2) which is UP in a 5x5 relative to 12
        ("Comida UP (Close)",   make_inputs(en=0.5, scents=[7])),
        ("Comida RIGHT (Far)",  make_inputs(en=0.5, scents=[14])), # (2,4) is 2 steps right
        ("Comida LEFT (Far)",   make_inputs(en=0.5, scents=[10])), # (2,0) is 2 steps left
    ]
    
    directions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    
    log_path = os.path.join(os.path.dirname(__file__), "inspection_log.txt")
    with open(log_path, "w", encoding='utf-8') as log:
        log.write(f"Best Genome ID: {best_genome.key} (Fitness: {best_genome.fitness})\n\n")
        
        for name, inputs in scenarios:
            outputs = net.activate(inputs)
            action_idx = outputs.index(max(outputs))
            chosen = directions[action_idx]
            out_fmt = [f"{x:.2f}" for x in outputs]
            
            log.write(f"Scenario: {name}\n")
            log.write(f"  Inputs: {len(inputs)} values\n")
            log.write(f"  Output: {out_fmt}\n")
            log.write(f"  Action: {chosen}\n\n")
            
    print(f"Log saved to {log_path}")

def main():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, CONFIG_FILE)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = load_latest_checkpoint()
    if not p: return

    best = get_best_genome(p)
    if not best:
        best = list(p.population.values())[0]
    
    print(f"BEST GENOME ID: {best.key} (Fitness: {best.fitness})")
    net = neat.nn.FeedForwardNetwork.create(best, config)
    simulate_brain(net, best)

if __name__ == "__main__":
    main()

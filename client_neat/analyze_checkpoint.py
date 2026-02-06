import neat
import pickle
import gzip
import sys
import os

# Import modules to ensure classes are available for unpickling
# We need to add the current directory to path to import client_neat if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import client_neat
except ImportError:
    print("⚠️ Could not import client_neat. Some custom classes might fail to unpickle.")

# Pickle needs the class to be defined in __main__ if it was saved from __main__
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

def analyze(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        print(f"❌ File not found: {checkpoint_file}")
        return

    print(f"📂 Loading: {checkpoint_file}...")
    
    try:
        with gzip.open(checkpoint_file) as f:
            population = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # Basic Stats
    print("\n📊 --- POPULATION STATS ---")
    print(f"Generation: {population.generation}")
    print(f"Population Size: {len(population.population)}")
    
    # Species
    if population.species:
        print(f"Species Count: {len(population.species.species)}")
    
    # User-readable Genome info
    best_genome = None
    if population.population:
        best_genome = max(population.population.values(), key=lambda g: g.fitness if g.fitness else -9999)
        
    if best_genome:
        print(f"\n🏆 --- BEST GENOME (ID: {best_genome.key}) ---")
        print(f"Fitness: {best_genome.fitness}")
        print(f"Nodes: {len(best_genome.nodes)}")
        print(f"Connections: {len(best_genome.connections)}")
        
        print("\n🧠 Brain Topology (Sample):")
        # Print a few connections
        i = 0
        for k, v in best_genome.connections.items():
            if v.enabled:
                print(f"   Input {k[0]} -> Node {k[1]} (Weight: {v.weight:.3f})")
                i += 1
                if i > 5:
                    print("   ... (more connections)")
                    break
    else:
        print("\n⚠️ No Genomes found in population?")

if __name__ == "__main__":
    target = "neat-checkpoint-continuous-auto"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    analyze(target)

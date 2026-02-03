import neat
import pickle
import os
import glob

# Config
CONFIG_FILE = "config-feedforward"

def load_latest_checkpoint():
    list_of_files = glob.glob('neat-checkpoint-*') 
    if not list_of_files:
        print("❌ Nenhum checkpoint encontrado.")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 Carregando checkpoint: {latest_file}")
    return neat.Checkpointer.restore_checkpoint(latest_file)

def get_best_genome(p):
    best = None
    best_fitness = -1.0
    for g_id, genome in p.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best = genome
    return best

def simulate_brain(net):
    print("\n🧠 --- SIMULAÇÃO DO CÉREBRO ---")
    
    # Scenarios: [Bias, En, St, S_U, S_D, S_L, S_R, W_U, W_D, W_L, W_R]
    scenarios = [
        ("Vazio (Energia Cheia)",  [1.0, 1.0, 0.0, 0,0,0,0, 0,0,0,0]),
        ("Parede CIMA",           [1.0, 0.8, 0.0, 0,0,0,0, 1,0,0,0]),
        ("Parede BAIXO",          [1.0, 0.8, 0.0, 0,0,0,0, 0,1,0,0]),
        ("Parede ESQ",            [1.0, 0.8, 0.0, 0,0,0,0, 0,0,1,0]),
        ("Parede DIR",            [1.0, 0.8, 0.0, 0,0,0,0, 0,0,0,1]),
        ("Comida CIMA",           [1.0, 0.5, 0.0, 1,0,0,0, 0,0,0,0]),
        ("Comida BAIXO",          [1.0, 0.5, 0.0, 0,1,0,0, 0,0,0,0]),
    ]
    
    directions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    
    for name, inputs in scenarios:
        outputs = net.activate(inputs)
        action_idx = outputs.index(max(outputs))
        action = directions[action_idx]
        print(f"📍 Cenário: {name:<20} -> Decisão: {action} (Out: {[f'{x:.2f}' for x in outputs]})")

def main():
    # Load Config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, CONFIG_FILE)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load Population
    p = load_latest_checkpoint()
    if not p: return

    # Get Best
    best = get_best_genome(p)
    if not best:
        print("⚠️ Checkpoint sem fitness registrado. Pegando o primeiro da lista.")
        best = list(p.population.values())[0]
    
    print(f"🏆 Melhor Genoma ID: {best.key} (Fitness: {best.fitness})")
    
    # Create Net
    net = neat.nn.FeedForwardNetwork.create(best, config)
    
    # Simulate
    simulate_brain(net)

if __name__ == "__main__":
    main()

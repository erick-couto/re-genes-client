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

def simulate_brain(net, best_genome):
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
        ("Comida ESQ",            [1.0, 0.5, 0.0, 0,0,1,0, 0,0,0,0]),
        ("Comida DIR",            [1.0, 0.5, 0.0, 0,0,0,1, 0,0,0,0]),
    ]
    
    directions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    
    with open("inspection_log.txt", "w") as log:
        log.write(f"Checkpoint Used\n")
        log.write(f"Best Genome ID: {best_genome.key} (Fitness: {best_genome.fitness})\n\n")
        
        for name, inputs in scenarios:
            outputs = net.activate(inputs)
            action_idx = outputs.index(max(outputs))
            
            chosen = directions[action_idx]
            
            out_fmt = [f"{x:.2f}" for x in outputs]
            log.write(f"Scenario: {name}\n")
            log.write(f"  Inputs: {inputs}\n")
            log.write(f"  Output: {out_fmt}\n")
            log.write(f"  Action: {chosen}\n\n")
            
    print("Log saved to inspection_log.txt")

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
    simulate_brain(net, best)

if __name__ == "__main__":
    main()

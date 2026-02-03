import asyncio
import websockets
import json
import neat
import os
import math
import pickle
import numpy as np

# --- CONFIGURAÇÃO ---
SERVER_URL = "wss://re-genes.is/ws/join?species=NEAT_Evo"
CONFIG_FILE = "config-feedforward"
CHECKPOINT_PREFIX = "neat-checkpoint-"

class NeatAmeba:
    def __init__(self, genome, config, genome_id):
        self.genome = genome
        self.config = config
        self.genome_id = genome_id
        
        # Cria a Rede Neural (FeedForward)
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Estado
        self.fitness = 0.0
        self.alive = True
        self.max_ticks = 2000 # Timeout para evitar loops infinitos de camping
        
        self.energy_gained = 0
        self.last_energy = 100
        
    async def run(self):
        """Vida de uma ameba"""
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                # Handshake
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                self.my_id = welcome_data.get("id")
                print(f"✅ [G{self.genome_id}] Connected as {self.my_id}")
                
                # Loop de Vida
                tick_count = 0
                while self.alive and tick_count < self.max_ticks:
                    try:
                        # print(f"[G{self.genome_id}] Waiting for msg...")
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        
                        if data['type'] == 'UPDATE':
                            if not data['alive']:
                                print(f"💀 [G{self.genome_id}] Died.")
                                self.alive = False
                            
                            # Atualiza energia para cálculo de fitness
                            current_energy = data.get('energy', 0)
                            delta = current_energy - self.last_energy
                            if delta > 0:
                                self.energy_gained += delta
                            self.last_energy = current_energy
                            continue
    
                        if data['type'] == 'TICK':
                            # Visão e Sensores
                            vision = data.get('vision') # 3 layers of 9x9
                            energy = data.get('energy', 0)
                            stomach = data.get('stomach', 0)
                            
                            # Processa Inputs
                            inputs = self.process_inputs(vision, energy, stomach)
                            
                            # Ativa Rede Neural
                            outputs = self.net.activate(inputs)
                            
                            # Escolhe Ação (Argmax)
                            # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
                            action_idx = outputs.index(max(outputs))
                            
                            cmd = "stay"
                            direction = "UP"
                            
                            if action_idx == 0: 
                                cmd, direction = "move", "UP"
                            elif action_idx == 1:
                                cmd, direction = "move", "DOWN"
                            elif action_idx == 2:
                                cmd, direction = "move", "LEFT"
                            elif action_idx == 3:
                                cmd, direction = "move", "RIGHT"
                            
                            # Envia
                            decision = {
                                "action": cmd,
                                "direction": direction
                            }
                            await websocket.send(json.dumps(decision))
                            
                            # Log detalhado para Debug (Genome 1)
                            if self.genome_id == 1 and tick_count % 20 == 0:
                               # Formata inputs para leitura fácil
                               # [Bias, En, St, S_U, S_D, S_L, S_R, W_U, W_D, W_L, W_R]
                               in_fmt = [f"{x:.1f}" for x in inputs]
                               out_fmt = [f"{x:.1f}" for x in outputs]
                               # print(f"[G1] T:{tick_count} IN:{in_fmt} OUT:{out_fmt} -> {direction}")
                            
                            tick_count += 1
                            
                    except websockets.exceptions.ConnectionClosed:
                        print(f"🔌 [Genome {self.genome_id}] Connection Closed.")
                        break
                    except Exception as e:
                        print(f"⚠️ [Genome {self.genome_id}] Runtime Error: {e}")
                        break
                        
                # Fim da Vida
                self.genome.fitness = tick_count + (self.energy_gained * 2)
                # print(f"💀 [Genome {self.genome_id}] Died. Fitness: {self.genome.fitness:.1f}")

        except Exception as e:
            print(f"❌ [Genome {self.genome_id}] Setup Error: {e}")
            self.genome.fitness = 0

    def process_inputs(self, vision, energy, stomach):
        """
        Transforma a matriz de visão bruta e status em 11 inputs para a NN.
        Inputs: Bias(1), Energy(1), Stomach(1), Scent(4), Wall(4)
        """
        # Se visão for nula (cego), retorna zeros
        # 3 layers of 9x9. If radius changes, this might break.
        if not vision or len(vision[0]) < 9: 
            return [0.0] * 11
        
        # Vision Layers: 0=Obstacles, 1=Scent, 2=Enemies
        # Grid 9x9. Centro é (4,4)
        
        # 1. Sensores de Obstáculos (Perto)
        # Verifica células adjacentes diretas
        # (y, x)
        wall_up    = 1.0 if vision[0][3][4] > 0 else 0.0
        wall_down  = 1.0 if vision[0][5][4] > 0 else 0.0
        wall_left  = 1.0 if vision[0][4][3] > 0 else 0.0
        wall_right = 1.0 if vision[0][4][5] > 0 else 0.0
        
        # 2. Sensores de Cheiro (Gradiente/Soma por quadrante ou direção?)
        scent_up    = vision[1][3][4]
        scent_down  = vision[1][5][4]
        scent_left  = vision[1][4][3]
        scent_right = vision[1][4][5]
        
        # Normalização de Energia (0 a 100+)
        norm_energy = min(energy, 200) / 200.0
        
        # Stomach (Agora temos via main.py update)
        norm_stomach = min(stomach, 50) / 50.0 
        
        bias = 1.0
        
        return [
            bias,
            norm_energy,
            norm_stomach,
            scent_up, scent_down, scent_left, scent_right,
            wall_up, wall_down, wall_left, wall_right
        ]

async def eval_genomes_async(genomes, config):
    """
    Roda a avaliação de todos os genomas em paralelo (AsyncIO)
    """
    tasks = []
    print(f"🚀 Iniciando Geração com {len(genomes)} amebas...")
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        ameba = NeatAmeba(genome, config, genome_id)
        tasks.append(ameba.run())
        
    await asyncio.gather(*tasks)
    
    # Stats da Geração
    fits = [g.fitness for _, g in genomes]
    print(f"📊 Geração Finalizada. Max: {max(fits):.1f}, Avg: {sum(fits)/len(fits):.1f}")
    
    # Dump do Melhor Cérebro
    best_genome = max(genomes, key=lambda x: x[1].fitness)[1]
    print(f"\n🧠 Best Genome Brain ({best_genome.key}):")
    # Imprime conexões (Input -> Output) relevantes
    for cg in best_genome.connections.values():
        if cg.enabled:
            print(f"   Node {cg.key[0]} -> {cg.key[1]}: w={cg.weight:.3f}")
    print("---------------------------------------------------\n")


def eval_genomes(genomes, config):
    """Wrapper síncrono para o NEAT chamar"""
    asyncio.run(eval_genomes_async(genomes, config))


def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, CONFIG_FILE)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Cria ou carrega população
    p = neat.Population(config)

    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix="neat-checkpoint-"))

    # Roda (x Gerações)
    winner = p.run(eval_genomes, 50)

    # Salva o vencedor
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    print(f'\n🏆 Best genome:\n{winner}')

if __name__ == '__main__':
    run_neat()

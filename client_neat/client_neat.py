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
                
                # Loop de Vida
                tick_count = 0
                while self.alive and tick_count < self.max_ticks:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    if data['type'] == 'UPDATE':
                        if not data['alive']:
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
                        
                        # Processa Inputs
                        inputs = self.process_inputs(vision, energy)
                        
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
                        await websocket.send(json.dumps({
                            "action": cmd,
                            "direction": direction
                        }))
                        
                        tick_count += 1
                        
                # Fim da Vida (Morte ou Timeout)
                # Fitness Function: Sobrevivencia + (Comida/Eficiência)
                self.genome.fitness = tick_count + (self.energy_gained * 2)
                # print(f"🧬 Genome {self.genome_id} finished. Fitness: {self.genome.fitness:.1f}")

        except Exception as e:
            print(f"Erro Genome {self.genome_id}: {e}")
            self.genome.fitness = 0

    def process_inputs(self, vision, energy):
        """
        Transforma a matriz de visão bruta e status em 11 inputs para a NN.
        Inputs: Bias(1), Energy(1), Stomach(1-TODO), Scent(4), Wall(4)
        """
        # Se visão for nula (cego), retorna zeros
        if not vision: return [0.0] * 11
        
        # Vision Layers: 0=Obstacles, 1=Scent, 2=Enemies
        # Grid 9x9. Centro é (4,4)
        
        # 1. Sensores de Obstáculos (Perto)
        # Verifica células adjacentes diretas
        # (y, x)
        wall_up    = vision[0][3][4] # (3,4) is UP form Center (4,4)
        wall_down  = vision[0][5][4]
        wall_left  = vision[0][4][3]
        wall_right = vision[0][4][5]
        
        # 2. Sensores de Cheiro (Gradiente/Soma por quadrante ou direção?)
        # Vamos pegar a média de intensidade em cada direção (Cone de visão)
        # Simples: Intensity at adjacents
        scent_up    = vision[1][3][4]
        scent_down  = vision[1][5][4]
        scent_left  = vision[1][4][3]
        scent_right = vision[1][4][5]
        
        # Normalização de Energia (0 a 200, mas pode ser mais. Normaliza por 100 ou 200)
        norm_energy = min(energy, 200) / 200.0
        
        # Stomach - Não tenho acesso via TICK yet (precisaria atualizar o servidor de novo pra mandar no TICK)
        # Por enquanto, assumir 0 ou tirar do input config?
        # Config pede 11 inputs. Vou mandar 0.0 por enquanto ou usar last_energy como proxy?
        # Vou mandar 0.0 placeholder para não quebrar a rede.
        norm_stomach = 0.0 # TODO: Update server to send stomach in TICK
        
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
    # p.add_reporter(neat.Checkpointer(5))

    # Roda (x Gerações)
    winner = p.run(eval_genomes, 50)

    # Salva o vencedor
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    print(f'\n🏆 Best genome:\n{winner}')

if __name__ == '__main__':
    run_neat()

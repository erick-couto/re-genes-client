import asyncio
import websockets
import json
import random
import os
import math

# --- CONFIGURAÇÃO DA ESPÉCIE MEMORIAM ---
SPECIES_NAME = "Memoriam-v1"
SERVER_URL = "wss://re-genes.is/ws/join?species=Memoriam"
SPECIES_NAME = "Memoriam-v1"
SERVER_URL = "wss://re-genes.is/ws/join?species=Memoriam"
Q_TABLE_BASE_NAME = "qtable_memoriam"

# Hiperparâmetros de Aprendizado
ALPHA = 0.1      # Taxa de Aprendizado (0.1 = aprende devagar e consistente)
GAMMA = 0.9      # Fator de Desconto (0.9 = valoriza o futuro)
EPSILON_START = 1.0  # Exploração Inicial (100% aleatório)
EPSILON_MIN = 0.05   # Exploração Mínima (5% aleatório)
EPSILON_DECAY = 0.995 # Decaimento por Tick

class MemoriamBrain:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON_START
        self.last_state = None
        self.last_action = None
        self.last_energy = 100
        
        # Carrega memória genética
        # Carrega memória genética
        self.species_desc = "Unknown"
        self.memory_file = f"{Q_TABLE_BASE_NAME}_Unknown.json"
        
    def set_phenotype(self, species_desc: str):
        """Define o fenótipo e carrega a memória apropriada."""
        # Sanitiza o nome (Gigante Lento -> Gigante_Lento)
        safe_name = species_desc.replace(" ", "_").replace("(", "").replace(")", "").strip()
        self.species_desc = safe_name
        self.memory_file = f"{Q_TABLE_BASE_NAME}_{safe_name}.json"
        print(f"🧬 Fenótipo detectado: {species_desc} -> Usando memória: {self.memory_file}")
        self.load_memory()

    def get_state_key(self, vision, energy):
        """
        Simplifica a visão matrix 3x3x3 em uma string de estado única.
        Foca apenas no CENTRO da visão (Raio 1) para reduzir complexidade.
        """
        if not vision:
            return "BLIND"
            
        # Visão é 3 canais (Obstacle, Scent, Enemy). Vision Radius do server é 4 (Matriz 9x9).
        # Vamos focar no crop central 3x3 (i=3 a 5)
        # Scent (Canal 1): Quantiza o cheiro em 3 níveis (Nada, Cheiroso, Muito Cheiroso)
        # Obstacle (Canal 0): Binário
        
        state_parts = []
        center_y, center_x = 4, 4 # Centro da matriz 9x9
        
        for y in range(center_y - 1, center_y + 2):
            for x in range(center_x - 1, center_x + 2):
                
                # Verifica Obstáculo (Canal 0)
                is_wall = 1 if vision[0][y][x] > 0 else 0
                
                # Verifica Cheiro (Canal 1) - Quantizado
                scent_val = vision[1][y][x]
                scent_lvl = 0
                if scent_val > 0.5: scent_lvl = 2
                elif scent_val > 0.1: scent_lvl = 1
                
                state_parts.append(f"{is_wall}{scent_lvl}")
                
        # Estado de Energia: Crítico (<20), Baixo (<50), Ok (>=50)
        energy_state = "OK"
        if energy < 20: energy_state = "CRIT"
        elif energy < 50: energy_state = "LOW"
        
        return f"{''.join(state_parts)}_{energy_state}"

    def choose_action(self, state_key):
        """Epsilon-Greedy Policy"""
        actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        
        if random.random() < self.epsilon:
            # Exploração (Aleatório)
            return random.choice(actions)
        else:
            # Exploração (Melhor Conhecido)
            if state_key not in self.q_table:
                return random.choice(actions)
            
            # Pega a ação com maior Q-Value
            q_values = self.q_table[state_key]
            best_action = max(q_values, key=q_values.get)
            return best_action

    def learn(self, current_energy):
        """Atualiza a Q-Table com base na recompensa"""
        if not self.last_state or not self.last_action:
            self.last_energy = current_energy
            return

        # --- CALCULA RECOMPENSA (Reward Function) ---
        reward = 0
        
        # 1. Delta de Energia (Comeu ou Gastou?)
        energy_delta = current_energy - self.last_energy
        
        if energy_delta > 0:
            reward += 50 # Comeu algo! Muito Bom!
        elif energy_delta < 0:
            reward -= 1 # Gastou energia (moveu ou ficou parado). Custo de vida.
            
        # 2. Punição por Morte (Energia zero)
        if current_energy <= 0:
            reward -= 100 # MORTE É RUIM!
            
        # --- ATUALIZAÇÃO Q-LEARNING (Bellman Equation) ---
        # Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        
        # Estado atual (mas não temos a visão atual aqui, ela vem no próximo tick decision)
        # Simplificação: Neste código, o learn() é chamado ANTES de decidir o próximo,
        # mas precisamos do "estado atual" para o max(Q(s')).
        # Como o learn é chamado quando recebemos o tick novo, já temos o estado novo lá fora.
        # Vamos ajustar a lógica no loop principal.
        pass 

    def update_q_value(self, reward, new_state_key):
        if self.last_state not in self.q_table:
            self.q_table[self.last_state] = {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]}
            
        if new_state_key not in self.q_table:
             self.q_table[new_state_key] = {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]}

        old_value = self.q_table[self.last_state][self.last_action]
        next_max = max(self.q_table[new_state_key].values())
        
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[self.last_state][self.last_action] = new_value

    def save_memory(self):
        if not self.memory_file: return
        
        try:
            # Atomic Write: Write to temp, then rename
            temp_file = self.memory_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.q_table, f)
            
            # Atomic replacement
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            os.rename(temp_file, self.memory_file)
            
            print(f"💾 Memória Salva ({self.species_desc}): {len(self.q_table)} estados.")
        except Exception as e:
            print(f"Erro ao salvar memória: {e}")

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.q_table = json.load(f)
                print(f"🧠 Memória Carregada ({self.species_desc}): {len(self.q_table)} estados.")
            except:
                print(f"🧠 Cérebro novo para {self.species_desc} (Memória vazia/corrompida).")
                self.q_table = {}
        else:
            print(f"🧠 Cérebro novo para {self.species_desc} (Primeira vez).")
            self.q_table = {}


async def viver_uma_vida(geracao, brain: MemoriamBrain):
    print(f"\n--- 🧠 Geração {geracao} (Epsilon: {brain.epsilon:.2f}) ---")
    print(f"🔌 Conectando ao servidor: {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            
            # --- FASE 1: NASCIMENTO ---
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            my_id = welcome_data.get("id")
            
            # Define fenótipo para carregar a memória correta
            species_desc = welcome_data.get("species", "Unknown").split("(")[0].strip() # Remove (Filha de...)
            brain.set_phenotype(species_desc)
            
            print(f"✅ Nasceu: {my_id} ({species_desc})")
            
            alive = True
            tick_vida = 0
            
            while alive:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data['type'] == 'UPDATE':
                    alive = data['alive']
                    current_energy = data.get('energy', 0)
                    
                    if not alive:
                        # Aprende com a morte
                        brain.update_q_value(-100, "DEATH")
                        print(f"💀 Morreu após {tick_vida} ticks.")
                        return 
                    continue 

                if data['type'] == 'TICK':
                    current_tick = data['tick']
                    vision = data.get("vision")
                    energy = data.get("energy", brain.last_energy) # Fallback if missing payload
                    
                    # 1. Percebe o Estado Atual
                    state_key = brain.get_state_key(vision, energy)
                    
                    # 2. Aprende com o Passado (Reward do que aconteceu entre o ultimo tick e agora)
                    # A recompensa é o delta de energia.
                    reward = 0
                    energy_delta = energy - brain.last_energy
                    if energy_delta > 0: reward = 50 
                    elif energy_delta == 0: reward = -0.1 # Leve punição por existir sem ganhar nada
                    else: reward = -1 # Punição normal por gasto de movimento
                    
                    if brain.last_state:
                         brain.update_q_value(reward, state_key)
                    
                    # 3. Decide Ação Futura
                    action_cmd = brain.choose_action(state_key)
                    
                    decision = {
                        "action": "move" if action_cmd != "STAY" else "stay",
                        "direction": action_cmd if action_cmd != "STAY" else "UP" # Direction doesn't matter for stay
                    }
                    
                    await websocket.send(json.dumps(decision))
                    
                    # 4. Atualiza Memória de Curto Prazo
                    brain.last_state = state_key
                    brain.last_action = action_cmd
                    brain.last_energy = energy
                    
                    if tick_vida % 10 == 0: 
                        print(f"Tick {current_tick} | Energy: {energy} | Action: {action_cmd} | Reward: {reward}")
                    tick_vida += 1
                    
    except Exception as e:
        print(f"⚠️ Erro: {e}")

async def ciclo_eterno():
    brain = MemoriamBrain()
    geracao = 1
    
    while True:
        await viver_uma_vida(geracao, brain)
        
        # Evolução e Persistência
        brain.save_memory()
        
        # Decaimento de Epsilon (Explorar menos, Exploitar mais)
        if brain.epsilon > EPSILON_MIN:
            brain.epsilon *= EPSILON_DECAY
            
        print("⏳ Reencarnando em 1 segundo...")
        await asyncio.sleep(1)
        geracao += 1

if __name__ == "__main__":
    try:
        asyncio.run(ciclo_eterno())
    except KeyboardInterrupt:
        print("\n🛑 Encerrando.")

import asyncio
import websockets
import json
import random
import time

# URL do Servidor (Já com WSS para HTTPS)
SERVER_URL = "wss://ameba.vindo.app/ws/join"

async def viver_uma_vida(geracao):
    """
    Representa o ciclo de vida de UMA ameba.
    Retorna quando ela morre ou a conexão cai.
    """
    print(f"\n--- 🧬 Iniciando Geração {geracao} ---")
    print(f"🔌 Conectando ao servidor: {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            
            # --- FASE 1: NASCIMENTO (Handshake) ---
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            my_id = welcome_data.get("id")
            
            print(f"✅ Nasceu! Nome de batismo: {my_id}")
            
            # --- FASE 2: SOBREVIVÊNCIA ---
            alive = True
            tick_vida = 0
            energy = 100 # Energia inicial padrão
            
            while alive:

                # 1. ESPERA O SINAL DO SERVIDOR (TICK ou UPDATE)
                message = await websocket.recv()
                data = json.loads(message)
                
                # Se for apenas confirmação de uma ação anterior
                if data['type'] == 'UPDATE':
                    alive = data['alive']
                    energy = data.get('energy', 0)
                    if not alive:
                        print(f"💀 [{my_id}] Morreu após {tick_vida} ticks.")
                        return # Fim da vida
                    continue # Volta para esperar o próximo TICK
                
                # Se for o SINAL DE TICK (Hora de Agir!)
                if data['type'] == 'TICK':
                    current_tick = data['tick']
                    
                    # 2. CÉREBRO (Toma decisão)
                    actions = ["move", "stay"]
                    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
                    
                    decision = {
                        "action": random.choice(actions),
                        "direction": random.choice(directions)
                    }

                    # 3. ENVIA AÇÃO (Reativo)
                    await websocket.send(json.dumps(decision))

                    if tick_vida % 10 == 0: 
                        print(f"[{my_id}] Tick {current_tick} | Energia: {energy}")
                    tick_vida += 1

                    
    except Exception as e:
        print(f"⚠️ Erro de conexão ou morte súbita: {e}")

async def ciclo_eterno():
    """
    Gerencia a reencarnação infinita.
    """
    geracao = 1
    while True:
        # Tenta viver uma vida
        await viver_uma_vida(geracao)
        
        # Intervalo entre vidas (para respirar e não floodar o server se cair)
        print("⏳ Reencarnando em 3 segundos...")
        await asyncio.sleep(3)
        geracao += 1

if __name__ == "__main__":
    try:
        asyncio.run(ciclo_eterno())
    except KeyboardInterrupt:
        print("\n🛑 Encerrando a simulação (CTRL+C detectado).")

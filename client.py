import asyncio
import websockets
import json
import random

SERVER_URL = "ws://localhost:8080/ws/ameba_python_01"

async def run_ameba():
    async with websockets.connect(SERVER_URL) as websocket:
        print("🧠 Cérebro conectado à Matriz!")
        
        last_processed_tick = -1 # Memória local do último tick visto
        
        while True:
            try:
                # 1. Recebe a visão do mundo
                message = await websocket.recv()
                world_state = json.loads(message)
                current_tick = world_state['tick']
                
                # Sincronismo: Só toma decisão se for um NOVO tick
                if current_tick > last_processed_tick:
                    
                    me = world_state["amebas"].get("ameba_python_01")
                    if me:
                        print(f"Tick {current_tick} | ({me['x']}, {me['y']}) | Energia: {me['energy']}")
                    
                    # Cérebro Aleatório
                    moves = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
                    chosen_move = random.choice(moves)
                    
                    action = {
                        "action": "move",
                        "direction": chosen_move
                    }
                    
                    await websocket.send(json.dumps(action))
                    
                    # Atualiza memória para não repetir ação neste tick
                    last_processed_tick = current_tick
                
                else:
                    # Se o tick é o mesmo, espera um pouco para não fritar a CPU
                    await asyncio.sleep(0.01)
                
            except websockets.exceptions.ConnectionClosed:
                print("Morte súbita (Desconectado).")
                break

if __name__ == "__main__":
    asyncio.run(run_ameba())

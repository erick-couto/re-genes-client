import asyncio
import websockets
import json
import os
import random
import ssl

SERVER_URL = "wss://re-genes.is/ws/join"


def _make_ssl_context():
    """TLS robusto (ver client_neat para detalhes). Usa o bundle do certifi porque o
    Windows store desta maquina tem um root vencido do Let's Encrypt (DST X3) que faz
    o OpenSSL falhar com 'certificate has expired'. Remove VERIFY_X509_STRICT (Avast).
    Env vars: REGENES_INSECURE_TLS=1 (desliga verificacao), REGENES_CA_EXTRA=ca.pem."""
    if os.getenv("REGENES_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("⚠️  REGENES_INSECURE_TLS=1 -> verificacao de certificado DESLIGADA")
        return ctx
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    extra = os.getenv("REGENES_CA_EXTRA")
    if extra and os.path.exists(extra):
        try:
            ctx.load_verify_locations(extra)
        except Exception:
            pass
    return ctx


SSL_CONTEXT = _make_ssl_context() if SERVER_URL.startswith("wss") else None

async def run_ameba():
    async with websockets.connect(SERVER_URL, ssl=SSL_CONTEXT) as websocket:
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

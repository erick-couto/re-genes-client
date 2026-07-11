import asyncio
import websockets
import json
import os
import random
import ssl
import time

# Server URL
SERVER_URL = "wss://re-genes.is/ws/join?species=Prokaryota"


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

async def live_a_life(generation):
    """
    Represents the life cycle of ONE ameba.
    Returns when it dies or the connection drops.
    """
    print(f"\n--- 🧬 Starting Generation {generation} ---")
    print(f"🔌 Connecting to server: {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL, ssl=SSL_CONTEXT) as websocket:
            
            # --- PHASE 1: BIRTH (Handshake) ---
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            my_id = welcome_data.get("id")
            
            print(f"✅ Born! Name: {my_id}")
            
            # --- PHASE 2: SURVIVAL ---
            alive = True
            life_ticks = 0
            energy = 100 # Standard initial energy
            
            while alive:

                # 1. WAIT FOR SERVER SIGNAL (TICK or UPDATE)
                message = await websocket.recv()
                data = json.loads(message)
                
                # If it's just a confirmation of a previous action
                if data['type'] == 'UPDATE':
                    alive = data['alive']
                    energy = data.get('energy', 0)
                    if not alive:
                        print(f"💀 [{my_id}] Died after {life_ticks} ticks.")
                        return # End of life
                    continue # Wait for next TICK
                
                # If it's TICK signal (Time to Act!)
                if data['type'] == 'TICK':
                    current_tick = data['tick']
                    
                    # 2. BRAIN (Make decision)
                    # Prokaryota species ignores vision and acts randomly
                    vision = data.get("vision") # Receives 3x3x3 Matrix (Obstacles, Scent, Enemies)
                    energy_sector = data.get("energy")

                    actions = ["move", "stay"]
                    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
                    
                    decision = {
                        "action": random.choice(actions),
                        "direction": random.choice(directions)
                    }

                    # 3. SEND ACTION (Reactive)
                    await websocket.send(json.dumps(decision))

                    if life_ticks % 10 == 0: 
                        vision_status = "Blind"
                        if vision:
                            # Try to read scent (Channel 1, Center 1,1)
                            center_scent = vision[1][1][1] 
                            vision_status = f"Scent={center_scent:.2f}"
                            
                        print(f"[{my_id}] Tick {current_tick} | Energy: {energy} | {vision_status}")
                    life_ticks += 1

                    
    except Exception as e:
        print(f"⚠️ Connection error or sudden death: {e}")

async def eternal_cycle():
    """
    Manages infinite reincarnation.
    """
    generation = 1
    while True:
        # Try to live a life
        await live_a_life(generation)
        
        # Interval between lives (to breathe and avoid flooding server)
        print("⏳ Reincarnating in 3 seconds...")
        await asyncio.sleep(3)
        generation += 1

if __name__ == "__main__":
    try:
        asyncio.run(eternal_cycle())
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation (CTRL+C detected).")

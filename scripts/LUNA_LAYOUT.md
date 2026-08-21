# Luna — mapa para qualquer IA (`ssh luna`, user `ai`)

> **Ops de 20/08/2026:** o streamer **não** é mais o container. Leia
> `D:\DESENVOLVIMENTO\regenes\LUNA_HANDOFF.md` (na Luna: `/home/ai/HANDOFF.md`).
> Este LAYOUT continua válido como árvore; a seção “streamer = docker” abaixo está
> **errada**.

Este arquivo é o índice. A cópia que os agentes no checkout Windows leem está em
`regenes/CLAUDE.md` §5 (workspace `D:\DESENVOLVIMENTO\regenes`).

## Árvore

```
/home/ai/
  LAYOUT.md                 ← você está aqui
  ameba-server/             mundo Docker (código + data/ da gênese local)
  regenes/
    world            → ../ameba-server
    client           Native NEAT + HyperNEAT (venv .venv)
    narrator         narrador (CPU) falando com Chatterbox HTTP
    streamer         Chromium+Xvfb+FFmpeg → YouTube
    chatterbox-ptbr  → ../chatterbox-ptbr
    f5tts-ptbr       → ../f5tts-ptbr
  chatterbox-ptbr/          TTS Chatterbox (GPU, venv próprio)
  f5tts-ptbr/               F5-TTS-pt-br (A/B; não é o narrador)
```

## O que está no ar (LAN)

| serviço | onde | como chega |
|---|---|---|
| mundo | container `ameba_world` | `http://192.168.1.12:8081/` (headless) |
| viewer | o mesmo processo, `/viewer` | `http://192.168.1.12:8081/viewer` |
| clientes → mundo | Native 20 + Hyper 20 | `ws://127.0.0.1:8081` (loopback) |
| Chatterbox TTS | `chatterbox_server.py` | `http://127.0.0.1:8765/synth` |
| narrador | `main.py` | spectate `ws://127.0.0.1:8081/ws/spectate`; PCM `:8790` |
| streamer | container `regenes_streamer` | viewer `127.0.0.1:8081` + áudio `:8790` → YouTube |
| RackNerd / Jenkins | **não** deploya isto | `ssh regenesis` é outra gênese |

YouTube: a live sai da Luna. A RackNerd não deve ter o streamer no ar ao mesmo tempo (mesma `STREAM_KEY`).

## Comandos

```bash
# mundo
cd /home/ai/ameba-server
docker compose -f docker-compose.yml -f docker-compose.luna.yml ps

# clientes Fase 2 (NÃO ligar legacy/)
/home/ai/regenes/client/scripts/start_luna.sh
/home/ai/regenes/client/scripts/stop_luna.sh

# narrador + Chatterbox (GPU)
/home/ai/regenes/narrator/scripts/start_luna.sh
/home/ai/regenes/narrator/scripts/stop_luna.sh
tail -f /home/ai/regenes/narrator/logs/narrator.log

# streamer (YouTube)
cd /home/ai/regenes/streamer
docker compose -f docker-compose.yml -f docker-compose.luna.yml --env-file .env ps
```

## TTS

- Receita: exaggeration=0.65 cfg=0.5 atempo=1.20
- Refs: `/home/ai/chatterbox-ptbr/refs/frozen/`
- F5 fora do narrador (acentuação infiel)

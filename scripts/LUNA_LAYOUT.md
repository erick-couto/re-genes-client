# Luna — mapa para qualquer IA (`ssh luna`, user `ai`)

> **Ops:** o streamer **não** é o container. Comandos abaixo; a história e as
> receitas NVENC estão em `LUNA_HANDOFF.md` (na Luna: `/home/ai/HANDOFF.md`).
> Mundo: compose só em `/home/ai/ameba-server` (não no symlink `regenes/world`).
> Client e ameba-server **não são git** — deploy por `scp`/`rsync`.
> **Start de cliente não empilha:** `start_luna.sh` para o trio velho (padrão
> `python -u host.py` / `host_hyper.py` / `host_grn.py`, não o caminho) e aborta
> se ainda houver processo. Esperado: 1 native + 1 hyper + 1 grn. Cap 60 no
> checkout do mundo; até o deploy, a Luna ainda é 50.

Este arquivo é o índice **na Luna**. No Windows: `AGENTS.md` + `CLAUDE.md` + `LUNA_HANDOFF.md`.

## Árvore

```
/home/ai/
  LAYOUT.md                 ← você está aqui
  HANDOFF.md                ops (errata no topo; diário 20/08 abaixo)
  ameba-server/             mundo Docker (código + data/ da gênese local)
  regenes/
    world            → ../ameba-server
    client           Native NEAT + HyperNEAT + GRN (venv .venv)
    narrator         narrador (CPU). TTS = TTS_ENGINE no .env (hoje: meça)
    streamer         Chromium+Xvfb+FFmpeg → YouTube
    chatterbox-ptbr  → ../chatterbox-ptbr
    f5tts-ptbr       → ../f5tts-ptbr
  chatterbox-ptbr/          TTS Chatterbox (GPU, venv próprio)
  f5tts-ptbr/               F5-TTS-pt-br (A/B; não é o narrador)
```

## O que está no ar (LAN)

| serviço | onde | como chega |
|---|---|---|
| mundo | container `ameba_world` | `http://192.168.1.10:8081/` (headless) |
| viewer | o mesmo processo, `/viewer` | `http://192.168.1.10:8081/viewer` |
| clientes → mundo | Native 20 + Hyper 20 + GRN 20 | `ws://127.0.0.1:8081` (loopback) |
| TTS | `grep TTS_ENGINE` no `.env` do narrador | Kokoro **ou** Chatterbox `:8765` |
| narrador | `main.py` | spectate `ws://127.0.0.1:8081/ws/spectate`; PCM `:8790` |
| streamer | processo host (Xvfb+Chrome+ffmpeg) | **não** é o container `regenes_streamer` |
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

# narrador (+ TTS conforme .env)
/home/ai/regenes/narrator/scripts/start_luna.sh
/home/ai/regenes/narrator/scripts/stop_luna.sh
tail -f /home/ai/regenes/narrator/logs/narrator.log

# streamer (YouTube) — processo HOST, não Docker. deploy_luna.sh do streamer QUEBRA o NVENC.
/home/ai/regenes/streamer/scripts/stop_luna.sh
/home/ai/regenes/streamer/scripts/start_luna.sh
# se o Studio ficar em "preparando": stop, esperar 10s, start
# detalhes: D:\DESENVOLVIMENTO\regenes\LUNA_HANDOFF.md  (na Luna: /home/ai/HANDOFF.md)
```

## TTS

- Meça `TTS_ENGINE` no `.env`. 24/08 na Luna: **kokoro**.
- Receita Chatterbox (se for esse o motor): exaggeration=0.65 cfg=0.5 atempo=1.20
- Refs Chatterbox: `/home/ai/chatterbox-ptbr/refs/frozen/`
- F5 fora do narrador (acentuação infiel)

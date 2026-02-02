# 🧬 re-genes.is: Neural Client

> **Neural Modules (Brains) for the Re-Genes simulation.**

## Sobre
Este repositório contém os agentes autônomos (Amebas) que habitam o servidor **re-genes-world**.

## Como Rodar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute um dos clientes (ex: Darwin, que evolui):
   ```bash
   python client_darwin.py
   ```

## Arquitetura
Os clientes operam em modo **Reativo**:
1. Conectam ao WebSocket do servidor.
2. Aguardam o sinal de `TICK`.
3. Processam a visão do mundo.
4. Enviam a decisão (Movimento).

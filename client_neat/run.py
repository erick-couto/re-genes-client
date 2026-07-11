"""Inicializador da espécie NEAT.  Uso: python run.py [N]  (default 8)."""
import os
import sys

# torna o SDK (regenes_agent.py, na pasta pai) e este diretório importáveis
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # pai -> regenes_agent
sys.path.insert(0, _HERE)                    # este dir -> client_neat / neat_agent

from regenes_agent import run
from neat_agent import make_factory

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    factory, pop = make_factory()
    try:
        run(factory, n)
    finally:
        print("💾 salvando checkpoint final…")
        pop.save_checkpoint()

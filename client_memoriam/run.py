"""Inicializador da espécie Memoriam (Q-table).  Uso: python run.py [N]  (default 8)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # pai -> regenes_agent
sys.path.insert(0, _HERE)                    # este dir -> client_memoriam / memoriam_agent

from regenes_agent import run
from memoriam_agent import make_factory, MemoriamAgent

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    try:
        run(make_factory(), n)
    finally:
        print("💾 salvando Q-tables…")
        MemoriamAgent.manager.save_all()

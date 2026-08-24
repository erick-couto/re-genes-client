"""Regra do juiz (empate saturado / 5-bis): o músculo só sorteia igualdade exata.

NULL_EPS permanece. A janela 0,05 com topo >= 0,9 era o que jogava fora a
ordem que o neurônio ainda tinha depois do tanh.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide_action import NULL_EPS, STAY, decide  # noqa: E402

# 0 frente, 1 trás, 2 esq, 3 dir, 4 fica, 5 ataca, 6 empurra
Z = [0.0] * 7


def _vec(*pairs):
    out = list(Z)
    for i, v in pairs:
        out[i] = v
    return out


def test_sem_sinal_fica():
    assert decide(Z) == STAY
    assert decide([0.01] * 7) == STAY
    almost = [NULL_EPS - 1e-9] * 7
    assert max(abs(x) for x in almost) < NULL_EPS
    assert decide(almost) == STAY


def test_vencedor_claro_e_argmax():
    assert decide(_vec((0, 0.4))) == 0
    assert decide(_vec((2, 0.2), (3, 0.8))) == 3
    assert decide(_vec((5, -0.9), (6, 0.06))) == 6


def test_ordem_estrita_no_topo_saturado_nao_sorteia():
    """O caso que o 0,05 destruía: tanh(2)=0,964 vs tanh(1,5)=0,905.

    Antigo: mx>=0,9 e margem 0,059 → ainda perto; com 0,95 vs 0,91 a
    janela dispara. Novo: sempre o maior.
    """
    out = _vec((0, 0.95), (2, 0.91), (4, 0.90))
    random.seed(0)
    got = {decide(out) for _ in range(200)}
    assert got == {0}


def test_margem_menor_que_005_ainda_respeita_ordem():
    out = _vec((3, 0.999), (5, 0.970))
    assert out[3] - out[5] < 0.05
    random.seed(1)
    got = {decide(out) for _ in range(100)}
    assert got == {3}


def test_empate_exato_sorteia_e_nao_cai_no_indice_zero():
    out = _vec((0, 1.0), (4, 1.0))
    random.seed(42)
    counts = {0: 0, 4: 0}
    for _ in range(400):
        a = decide(out)
        counts[a] += 1
    assert counts[0] > 80 and counts[4] > 80
    assert counts[0] + counts[4] == 400


def test_null_eps_vence_um_pico_abaixo_do_limiar():
    """Tudo ruído, um 0,04: ainda é sem sinal, não 'quase frente'."""
    assert decide(_vec((0, 0.04))) == STAY

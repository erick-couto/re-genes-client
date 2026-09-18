"""Juiz do efetor (#46 rev 17/09): NULL_EPS → STAY; senão softmax MOTOR_TEMP.

8 saídas: 0 frente, 1 trás, 2 esq, 3 dir, 4 fica, 5 ataca, 6 empurra, 7 bite.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide_action import BITE, MOTOR_TEMP, NULL_EPS, STAY, decide  # noqa: E402

N_OUT = 8
Z = [0.0] * N_OUT


def _vec(*pairs):
    out = list(Z)
    for i, v in pairs:
        out[i] = v
    return out


def _hist(out, n, seed):
    random.seed(seed)
    counts = [0] * len(out)
    for _ in range(n):
        counts[decide(out)] += 1
    return counts


def test_sem_sinal_fica():
    assert decide(Z) == STAY
    assert decide([0.01] * N_OUT) == STAY
    almost = [NULL_EPS - 1e-9] * N_OUT
    assert max(abs(x) for x in almost) < NULL_EPS
    assert decide(almost) == STAY


def test_null_eps_vence_um_pico_abaixo_do_limiar():
    """Tudo ruído, um 0,04: ainda é sem sinal, não 'quase frente'."""
    random.seed(0)
    for _ in range(50):
        assert decide(_vec((0, 0.04))) == STAY


def test_vencedor_claro_e_modal():
    """0,8 vs 0,2: a preferência pesa; T=0,08 deixa o modal ≥ ~95%."""
    out = _vec((0, 0.8), (2, 0.2))
    n = 2000
    counts = _hist(out, n, seed=0)
    assert counts[0] / n >= 0.95
    assert sum(counts) == n


def test_topo_quase_empatado_mistura():
    """O caso que a 5-bis tornava determinístico: 0,95 vs 0,91.

    Softmax mistura (os dois aparecem); o maior continua modal.
    Não é a loteria uniforme antiga (não exige ~50/50).
    """
    out = _vec((0, 0.95), (2, 0.91), (4, 0.90))
    n = 800
    counts = _hist(out, n, seed=1)
    seen = {i for i, c in enumerate(counts) if c > 0}
    assert 0 in seen and 2 in seen
    assert counts[0] > counts[2]
    assert counts[0] / n < 0.95  # se isto falhar, MOTOR_TEMP esfriou demais


def test_outs_flat_sem_vies_de_bite():
    """As 8 saídas iguais: bite não é especial no sampler."""
    out = [0.3] * N_OUT
    n = 8000
    counts = _hist(out, n, seed=7)
    frac_bite = counts[BITE] / n
    assert abs(frac_bite - 1.0 / N_OUT) < 0.03
    assert max(counts) / n < 0.18
    assert min(counts) / n > 0.08


def test_temperatura_documentada():
    assert MOTOR_TEMP == 0.08
    assert STAY == 4
    assert BITE == 7

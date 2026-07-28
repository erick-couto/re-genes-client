"""
test_operador.py — os testes do OPERADOR de crossover (§25).

A regra dos 75% do fork passou 72h invisível porque a suíte do mundo testa ROTEAMENTO
de herança (quem é entregue como pai) e o cliente — onde o cérebro é construído — não
tinha teste nenhum. Estes cobrem o mecanismo e o efeito, na especificação da contra-análise:

1. moeda honesta do bit enabled (com a regra dos 75% ativa daria 12,5% e falharia ruidoso);
2. taxa efetiva da poda bate com p=0,9 declarado (senão a meia-vida da Bíblia é ficção);
3. nó órfão não volta (fix #3 da v5.0);
4. efeito composto (slow): circuito fechado de 6 gerações — o bug custava só −3% por
   acasalamento, invisível em 1 cruzamento, mortal em 80 gerações.

Roda com: pytest client_native/tests/ -m "not slow" (rápidos) | -m slow (circuito)
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neat_brain as nb  # noqa: E402


def _par_com_discordancia():
    """Dois genomas primordiais com um gene homólogo garantido em discordância de enabled."""
    g1 = nb.random_genome(1)
    g2 = nb.random_genome(2)
    # garante ao menos um homólogo: copia uma inovação de g1 pra g2 com o bit invertido
    cg = next(iter(g1.connections.values()))
    copia = cg.copy()
    copia.enabled = not cg.enabled
    g2.connections[cg.key] = copia
    return g1, g2, cg.key


def test_moeda_honesta_do_bit_enabled():
    """Homólogo em discordância: o filho fica HABILITADO ~50% (regra dos 75% daria 12,5%)."""
    g1, g2, key = _par_com_discordancia()
    hab = 0
    N = 400
    for i in range(N):
        filho = nb.crossover(g1, g2, 1000 + i)
        if key in filho.connections and filho.connections[key].enabled:
            hab += 1
    frac = hab / N
    assert 0.35 < frac < 0.65, f"moeda viciada: {frac:.2f} habilitado (esperado ~0.50; regra dos 75% daria 0.125)"


def test_poda_respeita_p_declarado():
    """Silêncio COMPARTILHADO (desabilitado nos dois pais) sobrevive a ~90% por acasalamento."""
    g1 = nb.random_genome(3)
    g2 = nb.random_genome(4)
    # força 30 homólogos: copia conexões de g1 pra g2 e silencia nos DOIS (determinístico)
    chaves = list(g1.connections)[:30]
    for k in chaves:
        copia = g1.connections[k].copy()
        g2.connections[k] = copia
        g1.connections[k].enabled = False
        g2.connections[k].enabled = False
    sobreviveram = 0
    total = 0
    for i in range(120):
        filho = nb.crossover(g1, g2, 5000 + i)
        for k in chaves:
            total += 1
            if k in filho.connections:
                sobreviveram += 1
    taxa = sobreviveram / total
    assert 0.82 < taxa < 0.97, f"poda fora do declarado: sobrevivência {taxa:.2f} (esperado ~0.90)"


def test_sem_no_orfao():
    """Fix #3 (v5.0): todo nó do filho é saída ou exigido por uma conexão herdada."""
    g1 = nb.random_genome(5)
    g2 = nb.random_genome(6)
    nb.mutate(g1); nb.mutate(g2)
    filho = nb.crossover(g1, g2, 42)
    nb.mutate(filho)
    necessarios = {k for (i, o) in filho.connections for k in (i, o) if k >= 0}
    necessarios |= set(range(7))
    orfaos = set(filho.nodes) - necessarios
    assert not orfaos, f"nós órfãos no filho: {sorted(orfaos)[:5]}"


@pytest.mark.slow
def test_circuito_fechado_nao_derrete():
    """Efeito composto: K=6 gerações, pool de 24, fconns(g6) >= 95% de fconns(g0).
    Com a regra dos 75% a razão medida na janela v5.0 foi 0,73 — o bug era invisível
    num cruzamento (−3%) e mortal em 80 gerações. Este é o teste que a impede de voltar."""
    random.seed(7)
    pool = [nb.random_genome(i) for i in range(24)]

    def media_fconns(gs):
        return sum(nb.functional_complexity(g)[1] for g in gs) / len(gs)

    f0 = media_fconns(pool)
    for ger in range(6):
        filhos = []
        for i in range(23):                      # tamanho constante do pool
            a, b = random.sample(pool, 2)
            filho = nb.crossover(a, b, 9000 + ger * 100 + i)
            nb.mutate(filho)
            filhos.append(filho)
        pool = filhos + [nb.random_genome(100 + ger)]  # influxo primordial ~4%
    f6 = media_fconns(pool)
    razao = f6 / f0 if f0 else 1.0
    assert razao >= 0.95, f"cérebro derretendo sem ecologia: fconns {f0:.0f} -> {f6:.0f} (razão {razao:.2f})"

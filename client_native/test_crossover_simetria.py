"""
Testes da SIMETRIA DO CROSSOVER (§35).

O DEFEITO. Em `_sym_configure_crossover`, no empate de fitness:
  · gene exclusivo de parent1 entrava SEMPRE  (`cg1.copy()`)
  · gene exclusivo de parent2 entrava por MOEDA

E a fitness é igualada por construção (`crossover()` põe 1.0 nos dois), então TODO
acasalamento cai no empate. Pior: `genome1.fitness > genome2.fitness` é sempre falso,
logo `parent1` é sempre o SEGUNDO argumento — a catraca depende da ordem da chamada.

    E[filho] = comuns + 1,0·excl(p1) + 0,5·excl(p2) = n + 0,5·(n − c)

Medido no brain_bank de produção (150 pares, SEM mutação): filho = 117,8% da média dos
pais, contra 115,0% previstos e 100% do simétrico. Trocar a ordem mudava o tamanho em
79 de 80 pares.

O docstring de `crossover()` já prometia "cruzamento SIMÉTRICO". Agora é verdade.

TODOS OS TESTES RODAM SEM MUTAÇÃO. Foi a confusão que quase me fez atribuir a inflação ao
operador errado: `host.py` chama `crossover()` e depois `mutate()`, e a primeira medição
juntava os dois. Sem mutação: 117,8%. Com: 118,5%. A mutação contribui ~0,7 p.p.

Roda com:  pytest test_crossover_simetria.py   (ou: python test_crossover_simetria.py)
"""
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neat_brain as nb  # noqa: E402

N_REP = 300


def _genoma(chaves, seed=0):
    """Genoma artificial com um conjunto CONHECIDO de conexões (in, out)."""
    cfg = nb.load_config()
    g = nb.neat.DefaultGenome(seed)
    g.configure_new(cfg.genome_config)
    g.connections.clear()
    rng = random.Random(seed)
    for k in chaves:
        # inovação DETERMINÍSTICA por (in,out) — a mesma que o mundo usa; sem ela o
        # crossover não alinharia e o teste mediria outra coisa.
        cg = g.create_connection(cfg.genome_config, k[0], k[1],
                                 nb._det_innovation(k[0], k[1]))
        cg.weight = rng.uniform(-1, 1)
        cg.enabled = True
        g.connections[k] = cg
    return g


def _par(n_comum=40, n_excl=20):
    """Dois pais com interseção conhecida e tamanhos iguais."""
    comuns = [(-i, 0) for i in range(1, n_comum + 1)]
    ea = [(-(100 + i), 1) for i in range(n_excl)]
    eb = [(-(200 + i), 2) for i in range(n_excl)]
    return _genoma(comuns + ea, 1), _genoma(comuns + eb, 2), set(ea), set(eb)


def test_tamanho_do_filho_converge_a_media_parental():
    """O invariante central: sem mutação, E[filho] = (|A| + |B|) / 2."""
    a, b, _, _ = _par()
    media = (len(a.connections) + len(b.connections)) / 2
    tam = [len(nb.crossover(a, b, i).connections) for i in range(N_REP)]
    razao = st.mean(tam) / media
    assert abs(razao - 1.0) < 0.06, \
        f"filho = {razao:.1%} da média parental (o defeito dava ~117%)"


def test_ordem_dos_pais_nao_muda_a_distribuicao():
    """A catraca era dependente de qual argumento virava parent1."""
    a, b, _, _ = _par()
    ab = [len(nb.crossover(a, b, i).connections) for i in range(N_REP)]
    ba = [len(nb.crossover(b, a, i).connections) for i in range(N_REP)]
    d = abs(st.mean(ab) - st.mean(ba))
    assert d < 2.0, f"média {st.mean(ab):.1f} vs {st.mean(ba):.1f} — a ordem ainda importa"


def test_cada_gene_exclusivo_aparece_em_metade_dos_filhos():
    """Simetria fina: os dois lados têm de sortear com a mesma probabilidade."""
    a, b, ea, eb = _par()
    ca = cb = 0
    for i in range(N_REP):
        f = set(nb.crossover(a, b, i).connections)
        ca += len(f & ea); cb += len(f & eb)
    pa = ca / (N_REP * len(ea))
    pb = cb / (N_REP * len(eb))
    assert abs(pa - 0.5) < 0.07, f"exclusivos de A entram em {pa:.1%} (esperado ~50%)"
    assert abs(pb - 0.5) < 0.07, f"exclusivos de B entram em {pb:.1%} (esperado ~50%)"
    assert abs(pa - pb) < 0.07, f"assimetria persiste: A {pa:.1%} vs B {pb:.1%}"


def test_genes_comuns_sempre_entram():
    """A recombinação não pode perder o que os dois pais compartilham."""
    a, b, ea, eb = _par()
    comuns = set(a.connections) & set(b.connections)
    assert comuns, "premissa do teste: há genes comuns"
    for i in range(40):
        f = set(nb.crossover(a, b, i).connections)
        assert comuns <= f, f"perdeu {len(comuns - f)} genes comuns na semente {i}"


def test_pais_identicos_produzem_o_mesmo_tamanho():
    """Sem disjuntos não há sorteio: o filho é do tamanho dos pais."""
    a, _, _, _ = _par()
    b = _genoma(list(a.connections), 3)
    for i in range(20):
        assert len(nb.crossover(a, b, i).connections) == len(a.connections)


def test_nao_inflaciona_ao_longo_de_geracoes():
    """O efeito que motivou a §35: a catraca composta ao longo das gerações.

    1,178 por acasalamento levaria de 113 a 886 conexões em ~12 gerações — que é a ordem
    de grandeza medida em produção (886 conns, cérebro custando 1,18× o basal)."""
    a, b, _, _ = _par()
    tam0 = (len(a.connections) + len(b.connections)) / 2
    pais = [a, b]
    for ger in range(10):
        filhos = [nb.crossover(pais[0], pais[1], ger * 100 + k) for k in range(6)]
        filhos.sort(key=lambda g: len(g.connections))
        pais = [filhos[len(filhos)//2 - 1], filhos[len(filhos)//2]]   # os medianos
    tam10 = (len(pais[0].connections) + len(pais[1].connections)) / 2
    assert tam10 <= tam0 * 1.25, \
        f"10 gerações inflaram {tam0:.0f} -> {tam10:.0f} ({tam10/tam0:.2f}×)"


def test_sem_mutacao_no_caminho():
    """Guarda contra a confusão que quase inverteu a atribuição: estes testes medem o
    CROSSOVER isolado. Se alguém puser mutate() dentro de crossover(), isto pega."""
    a, b, _, _ = _par()
    antes = dict(a.connections)
    nb.crossover(a, b, 7)
    assert set(a.connections) == set(antes), "crossover não pode alterar os pais"


if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in testes:
        try:
            t(); print("PASS", t.__name__); ok += 1
        except AssertionError as e:
            print("FAIL", t.__name__, "->", e)
        except Exception as e:
            print("ERRO", t.__name__, "->", type(e).__name__, e)
    print(f"\n{ok}/{len(testes)} testes passaram.")
    sys.exit(0 if ok == len(testes) else 1)

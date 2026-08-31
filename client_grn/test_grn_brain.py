# -*- coding: utf-8 -*-
"""Contrato da sopa GRN: pack redondo, memória entre ticks, sem fitness."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grn_brain as gb  # noqa: E402


def test_pack_unpack_redondo():
    g = gb.random_genome(42)
    g2 = gb.unpack(gb.pack(g))
    assert g2.to_dict() == g.to_dict()


def test_primordial_e_magro():
    g = gb.random_genome(1)
    nodes, conns = gb.complexity(g)
    assert g.nh == gb.N_HID_BIRTH
    assert nodes == g.nh + gb.N_OUT
    assert 1 <= conns <= gb.N_REGS_BIRTH
    assert gb.genes_count(g) == g.nh + gb.N_OUT + len(g.regs)


def test_passo_devolve_8():
    g = gb.random_genome(7)
    soup = gb.Soup(g)
    out = soup.step([0.0] * gb.N_IN)
    assert len(out) == gb.N_OUT
    assert all(abs(x) <= 1.0001 for x in out)


def test_memoria_quebra_o_congelamento_feedforward():
    """Mesma observação agora, histórias diferentes → saídas diferentes.
    O Native feedforward não consegue isto; a sopa vaza concentração."""
    hid, out0 = gb.HID0, gb.OUT0
    g = gb.Genome(
        1,
        {hid: 0.0, **{out0 + i: 0.0 for i in range(gb.N_OUT)}},
        {hid: 0.15, **{out0 + i: 0.15 for i in range(gb.N_OUT)}},
        [[0, hid, 2.0, True], [hid, hid, 1.5, True], [hid, out0, 2.0, True]],
    )
    quiet = [0.0] * gb.N_IN
    pulse = [0.0] * gb.N_IN
    pulse[0] = 1.0
    a, b = gb.Soup(g), gb.Soup(g)
    for _ in range(8):
        a.step(pulse)
    out_a = a.step(quiet)
    out_b = b.step(quiet)
    assert abs(out_a[0] - out_b[0]) > 1e-3, (out_a[0], out_b[0])


def test_crossover_nao_inventa_especie():
    a = gb.random_genome(10)
    b = gb.random_genome(11)
    c = gb.crossover(a, b, 123)
    assert c.nh == max(a.nh, b.nh)
    keys_ab = {(r[0], r[1]) for r in a.regs} | {(r[0], r[1]) for r in b.regs}
    for r in c.regs:
        assert (r[0], r[1]) in keys_ab


def test_mutate_respeita_teto_de_ocultos():
    g = gb.random_genome(3)
    rng = __import__("random").Random(0)
    g.nh = gb.MAX_HIDDEN
    for i in range(gb.HID0, gb.HID0 + g.nh):
        g.bias.setdefault(i, 0.0)
        g.decay.setdefault(i, 0.25)
    g2 = gb.mutate(g, rng)
    assert g2.nh <= gb.MAX_HIDDEN


def test_ids_nao_colidem_com_saida_native():
    """Native usa saídas 0–7 (#72: +bocado). A sopa reserva 163–170 e esconde em 200+."""
    assert gb.OUT0 == gb.N_IN == 163
    assert gb.HID0 >= gb.OUT0 + gb.N_OUT
    assert gb.HID0 == 200


def test_funcional_nao_conta_aresta_que_nao_chega_na_saida():
    g = gb.Genome(1, {gb.HID0: 0.0, **{gb.OUT0 + i: 0.0 for i in range(gb.N_OUT)}},
                  {gb.HID0: 0.25, **{gb.OUT0 + i: 0.25 for i in range(gb.N_OUT)}},
                  [[0, gb.HID0, 1.0, True],          # input -> hidden (pode ser funcional)
                   [gb.HID0, gb.HID0 + 50, 1.0, True]])  # dst inexistente: ignora no alcance
    _fn, fc = gb.functional_complexity(g)
    # sem caminho hidden->output, a aresta 0->hidden não alcança saída
    assert fc == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        t()
        print("PASS", t.__name__)
        ok += 1
    print(f"{ok}/{len(tests)}")

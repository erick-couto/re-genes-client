"""
Testes do R-BLUR (§8 do Relatório Guardião, 04/08/2026).

Trava as propriedades obrigatórias da §8.1 e a que originou a correção: um sinal
presente SÓ à direita não pode aparecer à esquerda. Antes, aparecia — a convolução
rodava no índice serializado, e o índice 3 é (f=1,l=+1) enquanto o 4 é (f=2,l=-1):
a extrema direita de uma fileira encostava na extrema esquerda da seguinte.

Roda com:  pytest test_cone_psf.py   (ou: python test_cone_psf.py)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cone_psf                                                   # noqa: E402
from cone_psf import CONE_OFFSETS, psf, blur                      # noqa: E402

DIR = [i for i, (f, l) in enumerate(CONE_OFFSETS) if l > 0]
ESQ = [i for i, (f, l) in enumerate(CONE_OFFSETS) if l < 0]
EIXO = [i for i, (f, l) in enumerate(CONE_OFFSETS) if l == 0]
SIGMAS = [0.0, 0.3, 0.5, 1.0, 1.57, 2.5, 4.0, 6.0]


def _espelho(row):
    """Troca cada célula (f,l) por (f,-l): o cone espelhado."""
    idx = {c: i for i, c in enumerate(CONE_OFFSETS)}
    return [row[idx[(f, -l)]] for (f, l) in CONE_OFFSETS]


def test_cone_bate_com_o_do_mundo():
    """Se a geometria divergir de world.py:_build_cone, a PSF borra o cone errado."""
    esperado = [(0, 0)]
    for f in range(1, 7):
        w = min(3, (f + 1) // 2)
        for l in range(-w, w + 1):
            esperado.append((f, l))
    assert CONE_OFFSETS == esperado
    assert len(CONE_OFFSETS) == 31


def test_campo_constante_permanece_constante():
    for s in SIGMAS:
        out = blur([0.7] * 31, psf(s))
        assert all(abs(v - 0.7) < 1e-9 for v in out), f"sigma={s}: {out[:3]}"


def test_sigma_zero_e_identidade():
    row = [i / 31.0 for i in range(31)]
    assert blur(row, psf(0.0)) == row
    assert blur(row, psf(0.2)) == row       # abaixo do limiar de nitidez


def test_impulso_central_da_resposta_simetrica():
    """Impulso no eixo (l=0) tem de espalhar igual para os dois lados."""
    for s in (1.0, 2.5, 6.0):
        row = [0.0] * 31
        row[CONE_OFFSETS.index((3, 0))] = 1.0
        out = blur(row, psf(s))
        for (f, l), v in zip(CONE_OFFSETS, out):
            if l == 0:
                continue
            j = CONE_OFFSETS.index((f, -l))
            assert abs(v - out[j]) < 1e-9, f"sigma={s} assimétrico em {(f,l)}"


def test_espelhar_o_cone_espelha_a_saida():
    """Equivariância: blur(espelho(x)) == espelho(blur(x))."""
    row = [0.0] * 31
    for k in DIR:
        row[k] = 1.0
    for s in (0.5, 1.57, 4.0):
        a = blur(_espelho(row), psf(s))
        b = _espelho(blur(row, psf(s)))
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), f"sigma={s}"


def _blur_serial_antigo(row, sigma):
    """A implementação ANTIGA, preservada só para a regressão abaixo."""
    if sigma < 0.35:
        return list(row[:31])
    r = max(1, int(round(3.0 * sigma)))
    ker = [math.exp(-(d * d) / (2 * sigma * sigma)) for d in range(-r, r + 1)]
    s = sum(ker)
    ker = [w / s for w in ker]
    out = [0.0] * 31
    for k in range(31):
        acc = 0.0
        for j in range(len(ker)):
            idx = k + j - r
            idx = 0 if idx < 0 else (30 if idx > 30 else idx)
            acc += ker[j] * row[idx]
        out[k] = acc
    return out


def _contraste(out):
    return (sum(out[k] for k in DIR) / len(DIR)) - (sum(out[k] for k in ESQ) / len(ESQ))


def test_sinal_so_a_direita_nao_vaza_para_a_esquerda():
    """O DEFEITO QUE ORIGINOU A §8: o lado com sinal tem de ler mais alto, sempre."""
    row = [0.0] * 31
    for k in DIR:
        row[k] = 1.0
    # limiares abaixo do medido (05/08/2026), com margem — não são a asserção principal
    for s, minimo in ((1.00, 0.70), (1.57, 0.50), (2.50, 0.25), (4.00, 0.11)):
        assert _contraste(blur(row, psf(s))) >= minimo, f"sigma={s}"


def test_regressao_supera_a_convolucao_serial():
    """A ASSERÇÃO QUE TRAVA O PROPÓSITO DA §8: em TODA a faixa de acuidade observada em
    produção, a PSF geométrica preserva mais contraste lateral que a serial.

    Medido em 05/08 sobre os fconns do brain_bank (K=120, SIGMA_MAX=6):
        fconns  sigma   serial   geométrica   ganho
            60   4.00    0.081        0.130    1.6x
           338   1.57    0.292        0.550    1.9x   <- mediana do banco
           600   1.00    0.526        0.760    1.4x
    """
    row = [0.0] * 31
    for k in DIR:
        row[k] = 1.0
    for fconns in (60, 150, 270, 338, 433, 600):
        A = fconns / (fconns + 120.0)
        s = 6.0 * (1.0 - A)
        antigo = _contraste(_blur_serial_antigo(row, s))
        novo = _contraste(blur(row, psf(s)))
        assert novo > antigo, f"fconns={fconns}: {novo:.3f} <= {antigo:.3f}"
        assert novo / max(antigo, 1e-9) >= 1.3, \
            f"fconns={fconns}: ganho {novo/antigo:.2f}x abaixo do medido"


def test_monotonico_no_sigma():
    """Mais desfoque -> menos contraste lateral, sem inversões."""
    row = [0.0] * 31
    for k in DIR:
        row[k] = 1.0
    cs = []
    for s in (0.5, 1.0, 2.0, 4.0, 6.0):
        out = blur(row, psf(s))
        cs.append(sum(out[k] for k in DIR)/len(DIR) - sum(out[k] for k in ESQ)/len(ESQ))
    assert cs == sorted(cs, reverse=True), cs


def test_pesos_somam_um_e_nao_sao_negativos():
    for s in SIGMAS:
        for linha in psf(s):
            assert abs(sum(w for _, w in linha) - 1.0) < 1e-9
            assert all(w >= 0.0 for _, w in linha)


def test_determinismo():
    row = [(i * 7 % 13) / 13.0 for i in range(31)]
    for s in SIGMAS:
        assert blur(row, psf(s)) == blur(row, psf(s))


def test_shape_preservado():
    """198 entradas dependem de 6 canais × 31 células: o R-BLUR não muda shape."""
    for s in SIGMAS:
        assert len(blur([0.5] * 31, psf(s))) == 31


def test_os_dois_executores_usam_a_mesma_psf():
    """Native e HyperNEAT têm de produzir o mesmo encode para a mesma visão."""
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(raiz, "client_hyperneat"))
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    try:
        import host as hn
        import host_hyper as hh
    finally:
        sys.argv = argv
    vis = [[((i * 3 + ch) % 11) / 11.0 for i in range(31)] for ch in range(4)]
    qui = [[((i * 7 + ch) % 5) / 5.0 for i in range(9)] for ch in range(3)]
    for conns in (0, 60, 338, 600):
        a = hn.encode(vis, qui, 30.0, 5.0, 50.0, 50.0, 0.0, 1.0, hn.acuity_params(conns))
        b = hh.encode(vis, qui, 30.0, 5.0, 50.0, 50.0, 0.0, 1.0, hh.acuity_params(conns))
        assert len(a) == len(b) == 163
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b)), f"divergem em conns={conns}"


def test_canais_de_predacao_intactos_com_acuidade_baixa():
    """R5/#2 (card #2): o executor NÃO edita a observação. Com A=0 (cérebro sem conexões
    funcionais) os canais 2 (inimigo) e 3 (perigo) têm de sair do encode EXATAMENTE como a
    PSF os borra — o fade pred_w que os zerava para A<0,40 foi removido. Vale para os dois
    executores (paridade no tratamento dos canais)."""
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(raiz, "client_hyperneat"))
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    try:
        import host as hn
        import host_hyper as hh
    finally:
        sys.argv = argv
    vis = [[((i * 3 + ch) % 11) / 11.0 for i in range(31)] for ch in range(4)]
    qui = [[((i * 7 + ch) % 5) / 5.0 for i in range(9)] for ch in range(3)]
    for hx in (hn, hh):
        ac = hx.acuity_params(0)                    # A = 0 — o caso que o fade zerava por inteiro
        assert ac[2] == 0.0
        inp = hx.encode(vis, qui, 30.0, 5.0, 50.0, 50.0, 0.0, 1.0, ac)
        for ch in (2, 3):   # 52: perigo e comida — os dois ultimos do cone
            esperado = blur(vis[ch], ac[0])
            # 52 (#44): o vetor agora e [escalares][cone 4x31][quimico 3x9]. O cone comeca
            # depois dos escalares — DERIVADO, nunca cravado (foi assim que o painel do
            # viewer quebrou tres vezes).
            n_esc = len(inp) - 4 * 31 - 3 * 9
            obtido = inp[n_esc + ch * 31: n_esc + (ch + 1) * 31]
            assert all(abs(x - y) < 1e-12 for x, y in zip(obtido, esperado)), \
                f"{hx.__name__} canal {ch}: o executor ainda edita a observação"
            assert max(obtido) > 0.0, f"{hx.__name__} canal {ch} zerado (o velho fade)"


def test_custo_de_cpu():
    """A §7.2 do meu contraditório exigiu o custo antes de priorizar. Aqui está ele."""
    import time
    P = psf(1.57)                       # sigma do cérebro mediano do banco
    row = [0.3] * 31
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        blur(row, P)
    dt = time.perf_counter() - t0
    por_canal = dt / n
    # 6 canais por tick, 4 Hz, 50 clientes
    carga = por_canal * 6 * 4 * 50
    print(f"\n  blur: {por_canal*1e6:.1f} µs/canal | 6 canais × 4 Hz × 50 clientes = "
          f"{carga*100:.2f}% de um core")
    assert carga < 0.5, f"custo alto demais: {carga:.2%} de um core"


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

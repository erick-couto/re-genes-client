"""
Testes do R4/#3 (card #3, 08/2026): o escalar nº 3 do encode deixou de ser a endorfina
FABRICADA pelo executor (pico +100 por delta de energia, decaimento -0,2/tick, penalidade
de fome -2,0 — estado interno inventado) e passou a ser o FATO BRUTO `ingested` do mundo:
quanto entrou no estômago neste tick, normalizado pelo PRÓPRIO stomach_size — exatamente o
idioma que o encode já usava para damage/impact (§26). Sem estado: um tick não vaza para o
seguinte. Shape (198) e ordem são contrato e não mudam.

Roda com:  pytest test_escalar_ingested.py   (ou: python test_escalar_ingested.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(raiz, "client_hyperneat"))

_argv = sys.argv[:]          # host.py lê sys.argv no import; protege como no test_cone_psf
sys.argv = [_argv[0]]
try:
    import host as hn            # noqa: E402
    import host_hyper as hh      # noqa: E402
finally:
    sys.argv = _argv

HOSTS = (hn, hh)
# 52 (#44): o cone tem 4 canais (visao) e o quimico 3x9 (contato). VIS/QUI vem daqui
# para nenhum teste fixar a contagem a mao.
VIS = [[((i * 3 + ch) % 11) / 11.0 for i in range(31)] for ch in range(4)]
QUI = [[((i * 7 + ch) % 5) / 5.0 for i in range(9)] for ch in range(3)]


def _encode(hx, ingested, stomach_size=50.0, conns=338):
    return hx.encode(VIS, QUI, 30.0, 5.0, stomach_size, ingested, 0.0, 1.0,
                     hx.acuity_params(conns))


def test_ingested_normalizado_pelo_estomago():
    """Mesmo idioma de damage/impact: min(1.0, ingested / stomach_size)."""
    for hx in HOSTS:
        assert _encode(hx, 25.0)[3] == 0.5, hx.__name__
        assert _encode(hx, 10.0, stomach_size=200.0)[3] == 0.05, hx.__name__


def test_ingested_satura_em_um():
    """Comeu mais que o tanque -> 1.0 (clamp do idioma §26, não regra nova)."""
    for hx in HOSTS:
        assert _encode(hx, 100.0)[3] == 1.0, hx.__name__


def test_sem_o_campo_zero():
    """Fallback do TICK (obs.get("ingested", 0.0)): sem o campo, o slot é 0.0."""
    for hx in HOSTS:
        assert _encode(hx, 0.0)[3] == 0.0, hx.__name__


def test_nao_vaza_entre_ticks():
    """O slot é um fato DO TICK, não um estado: depois de um tick com ingested>0,
    o tick seguinte sem ingested tem de ler 0.0 (a endorfina decaía aos poucos;
    o fato bruto simplesmente não está mais lá)."""
    for hx in HOSTS:
        assert _encode(hx, 40.0)[3] == 0.8, hx.__name__
        assert _encode(hx, 0.0)[3] == 0.0, hx.__name__


def test_shape_163_preservado():
    for hx in HOSTS:
        assert len(_encode(hx, 25.0)) == 163, hx.__name__


def test_paridade_dos_dois_executores():
    """Native e HyperNEAT produzem o MESMO vetor para o mesmo tick (contrato do §15/§16)."""
    a = _encode(hn, 25.0)
    b = _encode(hh, 25.0)
    assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))


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

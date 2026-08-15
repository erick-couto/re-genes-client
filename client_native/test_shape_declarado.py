"""
§46 (R-SHAPE, card #38): o declarado no join BATE com o que o executor de fato fala.

O mundo passou a validar no join a declaração de contrato (protocol_version, n_obs,
n_actions). Estes testes garantem que a declaração dos dois executores de Fase 2
não é letra morta: N_OBS é o tamanho REAL do vetor do encode(), N_ACTIONS é o
tamanho REAL da tabela de ações, e os três valores vão na URL do join — exatamente
o que os fixtures vivos do mundo (re-genes-world/tests/test_shape_contract.py)
espelham. Se o shape mudar num host sem mudar no outro, a paridade (§15/§16) quebra
aqui antes de quebrar em produção.

Roda com:  pytest test_shape_declarado.py   (ou: python test_shape_declarado.py)
"""
import os
import sys
from urllib.parse import urlsplit, parse_qs

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
VIS = [[((i * 5 + ch) % 13) / 13.0 for i in range(31)] for ch in range(6)]


def _declarado(hx):
    """O que a URL do join declara, parseado."""
    q = parse_qs(urlsplit(hx.URL).query)
    return {k: q[k][0] for k in ("protocol_version", "n_obs", "n_actions")}


def _encode(hx):
    return hx.encode(VIS, 30.0, 5.0, 50.0, 12.5, 0.0, 1.0, hx.acuity_params(338))


def test_declaracao_presente_na_url():
    for hx in HOSTS:
        d = _declarado(hx)
        assert d["protocol_version"] == str(hx.PROTOCOL_VERSION) == "5", hx.__name__
        assert d["n_obs"] == str(hx.N_OBS) == "194", hx.__name__
        assert d["n_actions"] == str(hx.N_ACTIONS) == "7", hx.__name__


def test_n_obs_e_o_vetor_real():
    """O n_obs declarado é o que o encode() de fato monta — sem letra morta."""
    for hx in HOSTS:
        assert len(_encode(hx)) == hx.N_OBS == 194, hx.__name__


def test_n_actions_e_a_tabela_real():
    """O n_actions declarado é o tamanho da tabela que o decide() indexa."""
    for hx in HOSTS:
        assert len(hx.ACTIONS) == hx.N_ACTIONS == 7, hx.__name__
        assert [a["action"] for a in hx.ACTIONS] == [
            "forward", "backward", "turn", "turn", "stay", "attack", "push"], hx.__name__


def test_paridade_da_declaracao():
    """Os dois executores falam a MESMA língua (§15/§16): declarações idênticas."""
    assert _declarado(hn) == _declarado(hh)


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

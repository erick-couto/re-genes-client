"""
§46 (R-SHAPE, card #38): o declarado no join BATE com o que o executor de fato fala.

O mundo passou a validar no join a declaração de contrato (protocol_version, n_obs,
n_actions). Estes testes garantem que a declaração dos três executores de Fase 2
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
sys.path.insert(0, os.path.join(raiz, "client_grn"))

_argv = sys.argv[:]          # host.py lê sys.argv no import; protege como no test_cone_psf
sys.argv = [_argv[0]]
try:
    import host as hn            # noqa: E402
    import host_hyper as hh      # noqa: E402
    import host_grn as hg        # noqa: E402
finally:
    sys.argv = _argv

HOSTS = (hn, hh, hg)
# 52 (#44): o cone tem 4 canais (visao) e o quimico 3x9 (contato). VIS/QUI vem daqui
# para nenhum teste fixar a contagem a mao.
VIS = [[((i * 5 + ch) % 13) / 13.0 for i in range(31)] for ch in range(4)]
QUI = [[((i * 3 + ch) % 7) / 7.0 for i in range(9)] for ch in range(3)]


def _declarado(hx):
    """O que a URL do join declara, parseado."""
    q = parse_qs(urlsplit(hx.URL).query)
    return {k: q[k][0] for k in ("protocol_version", "n_obs", "n_actions")}


def _encode(hx):
    return hx.encode(VIS, QUI, 30.0, 5.0, 50.0, 12.5, 0.0, 1.0, hx.acuity_params(338))


def test_declaracao_presente_na_url():
    for hx in HOSTS:
        d = _declarado(hx)
        assert d["protocol_version"] == str(hx.PROTOCOL_VERSION) == "8", hx.__name__
        assert d["n_obs"] == str(hx.N_OBS) == "163", hx.__name__
        assert d["n_actions"] == str(hx.N_ACTIONS) == "8", hx.__name__


def test_n_obs_e_o_vetor_real():
    """O n_obs declarado é o que o encode() de fato monta — sem letra morta."""
    for hx in HOSTS:
        assert len(_encode(hx)) == hx.N_OBS == 163, hx.__name__


def test_n_actions_e_a_tabela_real():
    """O n_actions declarado é o tamanho da tabela que o decide() indexa."""
    for hx in HOSTS:
        assert len(hx.ACTIONS) == hx.N_ACTIONS == 8, hx.__name__
        assert [a["action"] for a in hx.ACTIONS] == [
            "forward", "backward", "turn", "turn", "stay", "attack", "push", "bite"], hx.__name__


def test_paridade_da_declaracao():
    """Os três executores falam a MESMA língua (§15/§16): declarações idênticas."""
    assert _declarado(hn) == _declarado(hh) == _declarado(hg)


def test_especie_no_join_e_distinta():
    """Isolamento (§15) começa no handshake: três rótulos, três pools."""
    def _sp(hx):
        return parse_qs(urlsplit(hx.URL).query)["species"][0]
    assert _sp(hn) == "Native_NEAT"
    assert _sp(hh) == "HyperNEAT"
    assert _sp(hg) == "GRN"
    assert parse_qs(urlsplit(hg.URL).query)["paradigm"][0] == "gene_regulatory_network"


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


def test_escalares_novos_do_43_entram_sem_normalizacao():
    """§50/§51 (#43): os quatro fatos novos (propriocepção do desfecho + pele) já chegam
    normalizados — bits em {0,1} e fração sobre 4 — então NÃO passam pelo estômago como
    damage/impact/ingested. Se alguém os dividir por `ss`, a pele de um corpo grande vira
    zero e o canal morre em silêncio, que é exatamente o defeito que o #41 achou na régua
    da comida."""
    for hx in HOSTS:
        ac = hx.acuity_params(60)
        vis = [[0.0] * 31 for _ in range(4)]
        qui = [[0.0] * 9 for _ in range(3)]
        inp = hx.encode(vis, qui, 30.0, 0.0, 500.0, 0.0, 0.0, 1.0, ac,   # estômago GRANDE de propósito
                        moved_self=1.0, moved_passive=1.0,
                        contact_body=0.75, contact_wall=0.25)
        assert inp[8] == 1.0, f"{hx.__name__}: moved_self deformado"
        assert inp[9] == 1.0, f"{hx.__name__}: moved_passive deformado"
        assert inp[10] == 0.75, f"{hx.__name__}: contact_body deformado"
        assert inp[11] == 0.25, f"{hx.__name__}: contact_wall deformado"


def test_escalares_novos_tem_default_zero():
    """Cliente que não recebeu os campos (mundo antigo) não pode explodir nem inventar sinal."""
    for hx in HOSTS:
        ac = hx.acuity_params(60)
        inp = hx.encode([[0.0] * 31 for _ in range(4)], [[0.0] * 9 for _ in range(3)],
                            30.0, 0.0, 50.0, 0.0, 0.0, 1.0, ac)
        assert inp[8:12] == [0.0, 0.0, 0.0, 0.0], f"{hx.__name__}: default nao e zero"
        assert len(inp) == 163

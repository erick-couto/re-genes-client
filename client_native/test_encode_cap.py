"""Captura do vetor de entrada REAL (#22/#26) — o que o cérebro viu, não o que uma réplica supõe.

POR QUE ESTE ARQUIVO EXISTE. O estudo `viabilidade_plasticidade.py` re-implementava a codificação
sensorial para replayar vidas offline. Essa réplica já divergiu do cliente uma vez, e o fecho do
#22 registra o estrago: "nove defeitos no instrumento, TODOS empurrando para reprovar", o pior
deles 93 das 194 entradas em zero por construção. E hoje ela está obsoleta de novo — monta 6
canais de cone (194 entradas) contra os 12 escalares + 4 canais de cone + 3 químicos de contato
(163) que o cliente monta.

Gravar o vetor de verdade elimina a classe inteira de erro: sem réplica não há divergência.

OS DOIS TESTES QUE MAIS IMPORTAM são `test_DESLIGADA_por_padrao` e
`test_falha_de_escrita_NAO_derruba_a_ameba`: instrumentação que liga sozinha ou que mata o
organismo ao falhar é pior que instrumentação nenhuma.
"""
import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _host(cap_path=None):
    """Recarrega o módulo para que `_CAP_PATH` seja relido do ambiente.

    `host.py` lê `sys.argv[1]` como número de amebas no import; sob pytest o argv carrega o
    caminho do teste e o import quebra com ValueError. Mesmo contorno de `test_cone_psf.py`.
    """
    if cap_path is None:
        os.environ.pop("ENCODE_CAP", None)
    else:
        os.environ["ENCODE_CAP"] = cap_path
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    try:
        import host
        return importlib.reload(host)
    finally:
        sys.argv = argv


def test_DESLIGADA_por_padrao():
    """Sem a variável de ambiente, não abre arquivo e não escreve nada."""
    h = _host(None)
    assert h._CAP_PATH is None
    h._cap_encode(1, "qualquer", [0.5] * 164)
    assert h._CAP_FH is None, "abriu arquivo sem ninguém pedir"
    assert h._CAP_N == 0


def test_grava_o_vetor_INTEIRO_com_id_e_tick():
    d = tempfile.mkdtemp()
    f = os.path.join(d, "cap.jsonl")
    h = _host(f)
    vet = [i / 1000.0 for i in range(164)]
    h._cap_encode(4242, "ameba_abc", vet)
    h._CAP_FH.flush()
    linha = json.loads(open(f, encoding="utf-8").read().strip())
    assert sorted(linha) == ["id", "t", "v"]
    assert linha["t"] == 4242 and linha["id"] == "ameba_abc"
    assert len(linha["v"]) == 164, "vetor truncado: %d" % len(linha["v"])
    for a, b in zip(linha["v"], vet):
        assert abs(a - b) < 1e-6, "o vetor gravado não é o que entrou"


def test_o_tamanho_do_vetor_bate_com_o_encode_de_verdade():
    """Se o número de entradas mudar, este teste falha ANTES de a captura sair errada."""
    h = _host(None)
    vis = [[0.0] * 31 for _ in range(4)]
    qui = [[0.0] * 9 for _ in range(3)]
    inp = h.encode(vis, qui, 1.0, 1.0, 10.0, 0.0, 0.0, 1.0, h.acuity_params(100))
    assert len(inp) == 164, \
        "o encode passou a devolver %d entradas — atualize a captura e o estudo" % len(inp)


def test_falha_de_escrita_NAO_derruba_a_ameba():
    """Instrumentação que mata o organismo ao falhar é pior que instrumentação nenhuma."""
    h = _host(os.path.join("/caminho", "que", "nao", "existe", "x.jsonl"))
    h._cap_encode(1, "y", [0.0] * 164)      # não pode levantar
    h._CAP_PATH = None

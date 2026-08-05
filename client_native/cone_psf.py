"""
cone_psf.py — R-BLUR: a PSF do olho vive na GEOMETRIA do cone, não no índice do buffer.

DEFEITO CORRIGIDO (§8 do Relatório Guardião, 04/08/2026)
--------------------------------------------------------
`_blur()` convolvia a lista linear dos 31 offsets:

    idx = k + j - r          # host.py / host_hyper.py

A serialização do cone empilha as fileiras — índice 3 é (f=1, l=+1) e índice 4 é
(f=2, l=-1). A extrema DIREITA de uma fileira encosta na extrema ESQUERDA da seguinte.
Com raio 5 sobre fileiras de 3 a 7 células, isso não é desfoque: é embaralhamento.

Uma PSF mistura pontos por DISTÂNCIA ESPACIAL, não por proximidade no buffer.
A serialização é implementação, não geometria (Goodman 2005; Marr 1982).

MAGNITUDE MEDIDA (05/08/2026, 177 genomas Native do brain_bank de produção)
---------------------------------------------------------------------------
Sinal puro de 1,0 nas células à direita, 0,0 à esquerda. Contraste antes do blur = 1,000.

    fconns   raio(lista)   contraste lateral que sobrava
        60        12           0,08
       338 (mediana)  5        0,29     <- o cérebro mediano perdia 71%
       433 (p90)      4        0,39
       600            3        0,53

Nenhum dos 177 genomas tinha raio 0. Consequência medida na arena:
  · `scent` é o canal MAIS influente dos seis (muda a ação em 66,7% dos casos, 177/177);
  · a resposta é UNIMODAL EM ZERO na direção lateral (69,5% dos genomas na faixa morta);
  · quimiotaxia realizada: G_geo IC95 [0,9865; 1,0806] — contém 1,0.
O cérebro reagia à INTENSIDADE do cheiro (que sobrevive ao blur) e não ao GRADIENTE
lateral (que não sobrevivia). Não havia sinal direcional para herdar nem para selecionar.

DECISÕES DE PROJETO (as duas perguntas deixadas para contraditório na §8.2)
---------------------------------------------------------------------------
1. Distância EUCLIDIANA, não geodésica. A oclusão pertence à FORMAÇÃO DA IMAGEM (o mundo
   já aplica line-of-sight ao montar o cone); o blur pertence ao APARATO SENSORIAL. Uma
   PSF geodésica significaria que a retina sabe onde estão as paredes — inferência vazando
   para dentro da transdução.
2. Obstáculos NÃO bloqueiam a PSF, pelo mesmo motivo: a oclusão já aconteceu upstream.

A LEI DE ACUIDADE NÃO MUDA: A = C/(C+K), sigma = SIGMA_MAX*(1-A), fixa no nascimento,
determinística, client-side. Muda só a MÉTRICA DE VIZINHANÇA — de "adjacente na lista"
para "adjacente no cone". Shape preservado: 31 células, mesma ordem, 194 entradas.
Não pede reset de banco.
"""
import math

# Cone egocêntrico, idêntico a world.py:_build_cone(). Offset (frente, lateral);
# +frente = para onde encara, +lateral = à direita dela.
def _build_cone():
    cells = [(0, 0)]
    for f in range(1, 7):
        w = min(3, (f + 1) // 2)
        for l in range(-w, w + 1):
            cells.append((f, l))
    return cells


CONE_OFFSETS = _build_cone()          # 31 offsets, ordem FIXA (a mesma do mundo)
N_CELLS = len(CONE_OFFSETS)
assert N_CELLS == 31

_PRUNE = 1e-6        # peso relativo abaixo disto não entra na soma esparsa
_cache = {}          # sigma arredondado -> PSF


def psf(sigma):
    """sigma (em CÉLULAS do cone) -> PSF esparsa: lista de 31 listas de (j, peso).

    Gaussiana isotrópica sobre a distância euclidiana no plano (frente, lateral),
    normalizada por linha (campo constante permanece constante). sigma pequeno -> identidade.
    Cacheada: a acuidade é fixa no nascimento, então cada valor é construído uma única vez.
    """
    key = round(float(sigma), 3)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    if key < 0.35:                                   # visão nítida: identidade exata
        out = [[(i, 1.0)] for i in range(N_CELLS)]
        _cache[key] = out
        return out
    dois_s2 = 2.0 * key * key
    out = []
    for i, (fi, li) in enumerate(CONE_OFFSETS):
        pesos = []
        for j, (fj, lj) in enumerate(CONE_OFFSETS):
            d2 = (fi - fj) ** 2 + (li - lj) ** 2
            pesos.append(math.exp(-d2 / dois_s2))
        s = sum(pesos)
        linha = [(j, w / s) for j, w in enumerate(pesos) if w / s >= _PRUNE]
        # renormaliza depois da poda: a soma tem de ser exatamente 1
        t = sum(w for _, w in linha)
        out.append([(j, w / t) for j, w in linha])
    _cache[key] = out
    return out


def blur(row, P):
    """Aplica a PSF a um canal de 31 células. Determinístico, sem aleatoriedade."""
    if len(row) < N_CELLS:
        row = list(row) + [0.0] * (N_CELLS - len(row))
    return [sum(w * row[j] for j, w in P[i]) for i in range(N_CELLS)]

"""
substrate.py — o CORPO GEOMÉTRICO do cérebro HyperNEAT.

A diferença central pro NEAT direto: aqui o cérebro SABE onde cada sensor mora.

No NEAT direto, as 161 entradas são independentes e sem relação entre si — pra evoluir
"comida à esquerda -> vira esquerda" a evolução precisa sortear ~31 fios caindo cada um na
ação certa com o sinal certo (medido no genoma real: comida->frente tem 3 fios, com 33 pesos
positivos e 44 NEGATIVOS — se anulam, efeito líquido ~0).

No HyperNEAT, um CPPN recebe as COORDENADAS de dois neurônios e devolve o peso entre eles.
A mesma regra vira UMA função suave: "peso(comida em (x,y) -> vira-esq em (-1,0)) alto quando
x<0". Uma expressão só, válida pras 31 células de uma vez.

FILOSOFIA: a geometria do cone já é FATO DO MUNDO (`CONE_OFFSETS` existe em world.py desde a
v3). Dar ao cérebro acesso à geometria do próprio corpo não vicia comportamento — é contar a
VERDADE sobre o corpo dele. Hoje a gente esconde essa verdade e o obriga a redescobrir por
sorteio. Ninguém escreve "vá pra comida": o CPPN nasce aleatório e a seleção decide.

Ref: Stanley, D'Ambrosio & Gauci (2009), "A Hypercube-Based Encoding for Evolving
Large-Scale Neural Networks", Artificial Life 15(2). CPPN: Stanley (2007).
"""
import math

# --- o cone (mesma geometria do mundo, world.py _build_cone) ---
def _build_cone():
    cells = [(0, 0)]                       # a própria célula (o que está sob ela)
    for f in range(1, 7):                  # 6 células à frente
        w = min(3, (f + 1) // 2)           # meia-largura cresce com a distância, teto 3
        for l in range(-w, w + 1):
            cells.append((f, l))
    return cells
CONE_OFFSETS = _build_cone()               # 31 células, mesma ORDEM do mundo

# canal -> profundidade z. Separa as 5 modalidades em "camadas" do hipercubo, então o CPPN
# pode tratar cheiro e obstáculo com regras diferentes (ou iguais, se a seleção preferir).
CHANNEL_Z = [-1.0, -0.5, 0.0, 0.5, 1.0]    # obstáculo, cheiro, inimigo, perigo, comida

# --- ENTRADAS (161, na MESMA ordem do encode do host) ---
INPUT_COORDS = []
# 6 escalares (bias, energia, estômago, endorfina, pace_sin, pace_cos): interocepção, não têm
# lugar no cone -> ficam numa fileira ATRÁS do corpo (y=-1.2), fora do campo visual.
for i in range(6):
    INPUT_COORDS.append((-1.0 + 2.0 * i / 5.0, -1.2, 0.0))
# 5 canais x 31 células do cone: x = lateral (esq<0, dir>0), y = distância à frente
for ch in range(5):
    for (f, l) in CONE_OFFSETS:
        INPUT_COORDS.append((l / 3.0, f / 6.0, CHANNEL_Z[ch]))
assert len(INPUT_COORDS) == 161, len(INPUT_COORDS)

# --- SAÍDAS (7): posicionadas pelo SIGNIFICADO DIRECIONAL da ação ---
# É isto que deixa a regra geométrica existir: "vira-esq" mora à esquerda (x=-1), então o CPPN
# pode expressar "sensor com x<0 -> peso alto pra saída com x<0" como uma função de x1*x2.
OUTPUT_COORDS = [
    (0.0,  1.0,  0.0),    # 0 frente    -> à frente
    (0.0, -1.0,  0.0),    # 1 trás      -> atrás
    (-1.0, 0.0,  0.0),    # 2 vira-esq  -> à esquerda
    (1.0,  0.0,  0.0),    # 3 vira-dir  -> à direita
    (0.0,  0.0,  0.0),    # 4 fica      -> no centro (não vai a lugar nenhum)
    (0.0,  1.0,  1.0),    # 5 ataca     -> age à frente, camada de contato (z=+1)
    (0.0,  1.0, -1.0),    # 6 empurra   -> age à frente, outra camada (z=-1)
]

# --- OCULTOS: grade 4x4 no plano do corpo. Substrato FIXO (HyperNEAT clássico): a topologia
# não cresce; quem ganha expressividade é o CPPN. É a troca do encoding indireto.
HIDDEN_COORDS = [(x, y, 0.0)
                 for x in (-0.6, -0.2, 0.2, 0.6)
                 for y in (-0.6, -0.2, 0.2, 0.6)]

N_IN, N_HID, N_OUT = len(INPUT_COORDS), len(HIDDEN_COORDS), len(OUTPUT_COORDS)

WEIGHT_SCALE = 3.0


def _scale(w: float) -> float:
    """CPPN cru (tanh, [-1,1]) -> peso da sinapse."""
    return WEIGHT_SCALE * max(-1.0, min(1.0, w))


def _query(cppn, c1, c2):
    """Pergunta ao CPPN os pesos entre dois pontos. A distância entra como entrada porque é o
    que deixa ele expressar LOCALIDADE ('só conecte o que está perto') sem ter que derivá-la."""
    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    return cppn.activate([c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], d])


def express(cppn):
    """Consulta o CPPN em cada par de coordenadas e PINTA a rede: (W_ih, W_ho, n_conns).

    É aqui que o encoding indireto acontece: o genoma (CPPN pequeno) vira uma rede grande.
    O custo é pago UMA vez, no nascimento.

    LEO (saída 2) decide SE cada conexão existe. Com o custo de cérebro (§15.3), isto deixa a
    ESPARSIDADE ser evoluível: carregar 2.5k sinapses custa 3.4× o metabolismo (morte em ~95
    ticks), então a seleção empurra o CPPN a expressar só o que vale. O tamanho do cérebro vira
    uma decisão ECONÔMICA da linhagem — não um número que a gente fixou no config.
    """
    W_ih = [[0.0] * N_IN for _ in range(N_HID)]
    W_ho = [[0.0] * N_HID for _ in range(N_OUT)]
    n = 0
    for h in range(N_HID):
        ch = HIDDEN_COORDS[h]
        for i in range(N_IN):
            out = _query(cppn, INPUT_COORDS[i], ch)
            if out[2] > 0.0:                       # LEO: a conexão existe?
                W_ih[h][i] = _scale(out[0])        # saída 0 = peso entrada->oculto
                n += 1
    for o in range(N_OUT):
        co = OUTPUT_COORDS[o]
        for h in range(N_HID):
            out = _query(cppn, HIDDEN_COORDS[h], co)
            if out[2] > 0.0:                       # LEO
                W_ho[o][h] = _scale(out[1])        # saída 1 = peso oculto->saída
                n += 1
    return W_ih, W_ho, n


def activate(W_ih, W_ho, inputs):
    """Forward pass do substrato: 161 -> 16 (tanh) -> 7 (tanh)."""
    hid = [math.tanh(sum(w * x for w, x in zip(W_ih[h], inputs) if w != 0.0))
           for h in range(N_HID)]
    return [math.tanh(sum(W_ho[o][h] * hid[h] for h in range(N_HID) if W_ho[o][h] != 0.0))
            for o in range(N_OUT)]

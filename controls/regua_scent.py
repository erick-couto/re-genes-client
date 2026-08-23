"""Política da régua de quimiotaxia — pura, sem I/O.

CONTROLE. Não aprende, não evolui, não reporta genoma.
Espécie de wire: ReguaScent. Ver controls/README.md e o card T7.

Contrato do mundo (protocolo v7 / §52):
  vision[4][31]  obstáculos, corpo, perigo, comida assinada (+planta / −carne)
  chemical[3][9] planta, carne, sangue — contato, coordenada de corpo
  7 ações egocêntricas (ACTION_SPEC)

O brief T7 falava em “cone 6 canais”. Esse sensor saiu no #44: cheiro não mora
mais no olho. A régua lê o que o organismo de fato recebe.

diet_gene NÃO vem no WELCOME nem no TICK. Inferimos por evidência de pasto:
  em comida com espaço no estômago e ingested=0 → aquele tipo não nutre;
  os dois canais começam liberados (onívoro, 75% da Luna).
"""
from __future__ import annotations

# índices — mesmos de world.py (CONE_OFFSETS / CHEM_OFFSETS / ACTION_SPEC)
CH_WALL, CH_BODY, CH_DANGER, CH_FOOD = 0, 1, 2, 3
CHEM_PLANT, CHEM_MEAT, CHEM_BLOOD = 0, 1, 2

# cone: (0,0), (1,-1), (1,0), (1,1), ...
CONE_UNDER = 0
CONE_AHEAD = 2          # (1, 0)
CONE_AHEAD_LEFT = 1     # (1,-1)
CONE_AHEAD_RIGHT = 3    # (1, 1)

# chemical 9: sob, F, FR, R, BR, B, BL, L, FL
CHEM_UNDER, CHEM_F, CHEM_FR, CHEM_R, CHEM_BR, CHEM_B, CHEM_BL, CHEM_L, CHEM_FL = range(9)

FWD, BACK, TURN_L, TURN_R, STAY, ATTACK, PUSH = range(7)

FOOD_EPS = 0.05
SCENT_EPS = 0.01
WALL_EPS = 0.5


class DietFilter:
    """Quais canais de cheiro/comida esta vida já sabe que nutrem."""

    def __init__(self):
        self.plant_ok = True
        self.meat_ok = True

    def update(self, food0: float, ingested: float, stomach: float, stomach_size: float):
        space = (stomach_size or 0.0) - (stomach or 0.0)
        if abs(food0) > FOOD_EPS and space > FOOD_EPS and ingested <= 0.0:
            if food0 > 0:
                self.plant_ok = False
            else:
                self.meat_ok = False
        if not self.plant_ok and not self.meat_ok:
            self.plant_ok = self.meat_ok = True

    def food_ok(self, food_val: float) -> bool:
        if food_val > FOOD_EPS:
            return self.plant_ok
        if food_val < -FOOD_EPS:
            return self.meat_ok
        return False

    def scent(self, plant, meat):
        out = [0.0] * 9
        for i in range(9):
            if self.plant_ok:
                out[i] += float(plant[i])
            if self.meat_ok:
                out[i] += float(meat[i])
        return out


def decide(vision, chemical, diet: DietFilter, rng) -> int:
    """Uma ação v7. Só direção e pasto; não ataca, não empurra, não aprende."""
    if (not vision or len(vision) < 4 or len(vision[0]) < 31
            or not chemical or len(chemical) < 2 or len(chemical[0]) < 9):
        return STAY

    walls = vision[CH_WALL]
    food = vision[CH_FOOD]
    scent = diet.scent(chemical[CHEM_PLANT], chemical[CHEM_MEAT])
    wall_ahead = walls[CONE_AHEAD] >= WALL_EPS

    if diet.food_ok(food[CONE_UNDER]):
        return STAY

    if diet.food_ok(food[CONE_AHEAD]) and not wall_ahead:
        return FWD
    if diet.food_ok(food[CONE_AHEAD_LEFT]):
        return TURN_L
    if diet.food_ok(food[CONE_AHEAD_RIGHT]):
        return TURN_R

    fwd = scent[CHEM_F] + 0.3 * scent[CHEM_FL] + 0.3 * scent[CHEM_FR]
    left = scent[CHEM_L] + 0.3 * scent[CHEM_FL] + 0.15 * scent[CHEM_BL]
    right = scent[CHEM_R] + 0.3 * scent[CHEM_FR] + 0.15 * scent[CHEM_BR]
    back = scent[CHEM_B]

    if wall_ahead:
        if left > right + SCENT_EPS:
            return TURN_L
        if right > left + SCENT_EPS:
            return TURN_R
        return TURN_L if rng.random() < 0.5 else TURN_R

    top = max(fwd, left, right, back)
    if top < SCENT_EPS:
        return FWD if rng.random() < 0.7 else (TURN_L if rng.random() < 0.5 else TURN_R)

    if back >= fwd and back >= left and back >= right and back > fwd + SCENT_EPS:
        return TURN_L if left >= right else TURN_R
    if fwd >= left and fwd >= right:
        return FWD
    return TURN_L if left >= right else TURN_R

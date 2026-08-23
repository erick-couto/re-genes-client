"""Régua de quimiotaxia — política pura, sem servidor."""
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regua_scent import (
    DietFilter, decide, FWD, TURN_L, TURN_R, STAY,
    CONE_UNDER, CONE_AHEAD, CONE_AHEAD_LEFT, CONE_AHEAD_RIGHT,
)


def _blank():
    vis = [[0.0] * 31 for _ in range(4)]
    chem = [[0.0] * 9 for _ in range(3)]
    return vis, chem


def test_stay_on_digestible_food():
    vis, chem = _blank()
    vis[3][CONE_UNDER] = 0.8
    assert decide(vis, chem, DietFilter(), random.Random(0)) == STAY


def test_leave_indigestible_after_failed_graze():
    diet = DietFilter()
    diet.update(food0=0.8, ingested=0.0, stomach=0.0, stomach_size=20.0)
    assert diet.plant_ok is False
    vis, chem = _blank()
    vis[3][CONE_UNDER] = 0.8
    chem[1][1] = 0.9  # carne à frente
    assert decide(vis, chem, diet, random.Random(0)) == FWD


def test_forward_to_food_ahead():
    vis, chem = _blank()
    vis[3][CONE_AHEAD] = 0.6
    assert decide(vis, chem, DietFilter(), random.Random(0)) == FWD


def test_turn_toward_food_on_flank():
    vis, chem = _blank()
    vis[3][CONE_AHEAD_LEFT] = 0.6
    assert decide(vis, chem, DietFilter(), random.Random(0)) == TURN_L
    vis, chem = _blank()
    vis[3][CONE_AHEAD_RIGHT] = 0.6
    assert decide(vis, chem, DietFilter(), random.Random(0)) == TURN_R


def test_wall_ahead_turns_to_scent():
    vis, chem = _blank()
    vis[0][CONE_AHEAD] = 1.0
    chem[0][7] = 0.8  # planta à esquerda
    chem[0][3] = 0.1
    assert decide(vis, chem, DietFilter(), random.Random(1)) == TURN_L


def test_follow_front_scent():
    vis, chem = _blank()
    chem[0][1] = 0.7
    assert decide(vis, chem, DietFilter(), random.Random(0)) == FWD


def test_broken_obs_stays():
    assert decide(None, None, DietFilter(), random.Random(0)) == STAY


def test_omnivore_keeps_both_channels_after_a_meal():
    diet = DietFilter()
    diet.update(food0=0.8, ingested=3.0, stomach=5.0, stomach_size=20.0)
    assert diet.plant_ok and diet.meat_ok


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("todos passaram")

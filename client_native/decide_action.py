"""Contrato do efetor: 7 saídas da rede -> índice da ação. Native e HyperNEAT.

NULL_EPS: nervo desconectado não dispara músculo (fica). Justificativa própria,
medida: cérebro-zero ganhava "frente" de graça pelo argmax do índice 0.

Empate: sorteio uniforme somente quando o topo é exatamente igual. Qualquer
ordem estrita — inclusive a que o tanh comprime para margem < 0,05 — é
respeitada. Não é dial: remove o limiar absoluto 0,05 do "empate saturado".

Card: empate saturado / 5-bis (regra do juiz). Não calibra eat-rate.
"""
import random

NULL_EPS = 0.05
STAY = 4


def decide(out):
    mx = max(out)
    if max(abs(mx), abs(min(out))) < NULL_EPS:
        return STAY
    tied = [i for i, v in enumerate(out) if v == mx]
    if len(tied) > 1:
        return random.choice(tied)
    return tied[0]

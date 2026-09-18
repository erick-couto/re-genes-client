"""Contrato do efetor: 8 saídas da rede -> índice da ação. Native, HyperNEAT e GRN.

(#72: a 8ª saída é o bocado — comer virou ação; a regra do juiz não ensina a morder.)

NULL_EPS: nervo desconectado não dispara músculo (fica). Justificativa própria,
medida: cérebro-zero ganhava "frente" de graça pelo argmax do índice 0.

MOTOR_TEMP: o músculo não é um relé digital. Com sinal acima do nulo, a ação é
amostrada por softmax de temperatura baixa — vencedor-leva-tudo com vazamento,
ponderado pela intensidade. Margem grande quase sempre vence; margem pequena
mistura. Não é roleta uniforme. Não é exploração dirigida. Não há viés de bite.

Isto REABRE o #46 no ponto que a 5-bis fechou (ordem estrita depois do tanh
sempre respeitada). O que a 5-bis matou — loteria UNIFORME na janela 0,05 com
mx ≥ 0,9, que jogava fora ordem medida — continua morto. Softmax não é aquela
janela: 0,95 vs 0,91 não viram moeda 50/50.

Analogia (não a equação):
  Harris & Wolpert 1998, Nature — doi:10.1038/29528
  Faisal, Selen & Wolpert 2008, Nat Rev Neurosci — doi:10.1038/nrn2258
O paper fala tremor no MESMO comando; aqui o vazamento é entre pools motores
discretos. MOTOR_TEMP = 0,08 é coeficiente de lei na escala tanh ~[-1, 1],
escolhido para margem decisiva ficar quase determinística e margem miúda
misturar. Não retunar contra eat-rate nem spin (#35).

Card: https://github.com/erick-couto/re-genes-analysis/issues/46
"""
import math
import random

NULL_EPS = 0.05
STAY = 4
MOTOR_TEMP = 0.08
BITE = 7


def decide(out):
    mx = max(out)
    if max(abs(mx), abs(min(out))) < NULL_EPS:
        return STAY
    # estável: um peso é sempre exp(0) = 1
    weights = [math.exp((v - mx) / MOTOR_TEMP) for v in out]
    return random.choices(range(len(out)), weights=weights, k=1)[0]

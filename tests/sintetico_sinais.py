"""Gerador de sinais SINTETICOS para testar o pipeline sem camera.

Cada "pessoa" tem corpo em posicao/tamanho diferentes, velocidade diferente e
jeito ligeiramente diferente de fazer o sinal. Nao substitui gravacao real --
serve para garantir que features + DTW + segmentacao fazem o que deveriam.
"""
from __future__ import annotations

import numpy as np

from core_ai.corpo_tracker import (BOCA_D, BOCA_E, NARIZ, OLHO_D, OLHO_E, OMBRO_D, OMBRO_E,
                                   ORELHA_D, ORELHA_E, QuadroCorpo)

W, H = 1280, 720


def _mao(forma: str, rng) -> np.ndarray:
    """21 pontos relativos ao pulso (unidade ~ tamanho da mao), y para baixo."""
    base = np.zeros((21, 3), np.float32)
    dedos = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    angulos = np.deg2rad([-60, -20, 0, 20, 40])
    estendido = {"aberta": [1, 1, 1, 1, 1], "c": [.7, .7, .7, .7, .7], "indicador": [0, 1, 0, 0, 0],
                 "o": [.4, .4, .3, .3, .3]}[forma]
    for cadeia, ang, ext in zip(dedos, angulos, estendido):
        for k, idx in enumerate(cadeia):
            r = 0.35 + 0.25 * k * (0.3 + 0.7 * ext)
            curva = (1 - ext) * 0.8 * k
            base[idx, 0] = r * np.sin(ang) + curva * 0.1
            base[idx, 1] = -r * np.cos(ang) + curva * 0.15
            base[idx, 2] = -curva * 0.1
    return base + rng.normal(0, 0.01, base.shape).astype(np.float32)


def _pose(cx, cy, larg) -> np.ndarray:
    p = np.zeros((33, 3), np.float32)
    p[OMBRO_E, :2], p[OMBRO_D, :2] = [cx + larg / 2, cy], [cx - larg / 2, cy]
    fy = cy - larg * 0.95
    p[NARIZ, :2] = [cx, fy]
    p[OLHO_E, :2], p[OLHO_D, :2] = [cx + larg * .09, fy - larg * .08], [cx - larg * .09, fy - larg * .08]
    p[BOCA_E, :2], p[BOCA_D, :2] = [cx + larg * .07, fy + larg * .12], [cx - larg * .07, fy + larg * .12]
    p[ORELHA_E, :2], p[ORELHA_D, :2] = [cx + larg * .22, fy], [cx - larg * .22, fy]
    return p


# Trajetorias: funcao t in [0,1] -> {lado: (alvo_ponta_indicador em coords do corpo, forma)}
# coords do corpo: origem meio dos ombros, unidade = largura do ombro, y para baixo.
# A mao DIREITA (dominante) aparece do lado esquerdo da imagem espelhada (x negativo).
def _olho(lado):    return np.array([-0.09 if lado == "Right" else 0.09, -1.03])
def _bochecha(l):   return np.array([-0.15 if l == "Right" else 0.15, -0.9])


SINAIS = {
    "OCULOS": lambda t: {"Right": (_olho("Right") + [0, 0.02 * np.sin(6 * t)], "c"),
                         "Left": (_olho("Left") + [0, 0.02 * np.sin(6 * t)], "c")},
    "SAUDADE": lambda t: {"Right": (np.array([0.05 + .12 * np.cos(2 * np.pi * t),
                                              0.35 + .12 * np.sin(2 * np.pi * t)]), "aberta")},
    "TCHAU": lambda t: {"Right": (np.array([-0.5 + 0.15 * np.sin(4 * np.pi * t), -1.0]), "aberta")},
    "COVINHA": lambda t: {"Right": (_bochecha("Right") + [0.01 * np.sin(8 * t), 0], "indicador")},
    # sinal-nome composto: oculos -> cabelo cacheado (espiral do lado da cabeca) -> covinha
    "NOME_IRMA": lambda t: (
        {"Right": (_olho("Right"), "c")} if t < 0.33 else
        {"Right": (np.array([-0.3 + .06 * np.cos(12 * t), -1.25 + .06 * np.sin(12 * t)]), "indicador")}
        if t < 0.66 else {"Right": (_bochecha("Right"), "indicador")}),
}


def gerar(sinal: str, pessoa: int, repeticao: int, fps: float = 30.0,
          repouso: bool = False) -> list[QuadroCorpo]:
    rng = np.random.default_rng(1000 * pessoa + repeticao + sum(map(ord, sinal)) % 997)
    prng = np.random.default_rng(pessoa)        # caracteristicas FIXAS da pessoa
    larg = prng.uniform(260, 420)
    cx, cy = W / 2 + prng.uniform(-120, 120), H * prng.uniform(0.75, 0.95)
    duracao = prng.uniform(0.8, 1.6) * rng.uniform(0.9, 1.1)
    jeito = prng.normal(0, 0.02, 2)              # cada um faz um pouco deslocado
    tam_mao = larg * prng.uniform(0.32, 0.4)
    pose = _pose(cx, cy, larg)

    n = int(duracao * fps)
    quadros = []
    t0 = rng.integers(0, 10_000)
    linha_repouso = [np.array([-0.3, 1.4]), np.array([0.3, 1.4])]
    extra = int(0.5 * fps) if repouso else 0
    for i in range(-extra, n + extra):
        t = min(max(i / max(n - 1, 1), 0), 1)
        alvos = SINAIS[sinal](t) if 0 <= i < n else \
            {"Right": (linha_repouso[0], "aberta"), "Left": (linha_repouso[1], "aberta")}
        maos = {}
        for lado, (alvo, forma) in alvos.items():
            alvo = alvo + jeito + rng.normal(0, 0.008, 2)
            pts = _mao(forma, rng) * tam_mao
            if lado == "Left":
                pts[:, 0] = -pts[:, 0]
            alvo_px = np.array([cx, cy]) + alvo * larg
            pts[:, :2] += alvo_px - pts[8, :2]
            maos[lado] = pts.astype(np.float32)
        p = pose + rng.normal(0, 1.0, pose.shape).astype(np.float32)
        quadros.append(QuadroCorpo(maos=maos, pose=p, img_w=W, img_h=H,
                                   ts_ms=int(t0 + (i + extra) * 1000 / fps)))
    return quadros

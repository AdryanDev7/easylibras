"""
Normalizacao dos landmarks e construcao do vetor de features.

Este arquivo e a UNICA fonte da verdade. Coleta, treino, teste ao vivo e API
chamam exatamente a mesma funcao. Se voce mudar a normalizacao, MUDE A VERSAO
e retreine -- senao o modelo recebe features diferentes das que aprendeu
(e o erro e silencioso: acuracia boa no treino, lixo na camera).
"""
from __future__ import annotations

import numpy as np

VERSAO_FEATURES = 1

PULSO = 0
MEDIO_MCP = 9
PONTAS = [4, 8, 12, 16, 20]  # polegar, indicador, medio, anelar, minimo

DIM_FEATURES = 21 * 3 + 10 + 5  # 63 coords + 10 dist entre pontas + 5 ponta->pulso


def normalizar_pontos(pontos: np.ndarray, lado: str) -> np.ndarray:
    """(21,3) em pixels -> (21,3) invariante a posicao, tamanho e mao usada.

    1. Espelha mao esquerda para o referencial da direita (1 modelo serve as duas).
    2. Translada: pulso vai para a origem.
    3. Escala: divide pela distancia pulso -> base do dedo medio.

    NAO normalizamos rotacao de proposito: em Libras a orientacao da mao
    distingue letras (ex.: G e H), entao girar tudo apagaria informacao.
    """
    pts = np.asarray(pontos, dtype=np.float32).reshape(21, 3).copy()

    if lado and lado.lower().startswith("l"):
        pts[:, 0] = -pts[:, 0]

    pts -= pts[PULSO]

    escala = float(np.linalg.norm(pts[MEDIO_MCP]))
    if escala < 1e-6:
        escala = 1e-6
    pts /= escala
    return pts


def montar_features(pontos: np.ndarray, lado: str) -> np.ndarray:
    """Vetor final entregue ao classificador."""
    pts = normalizar_pontos(pontos, lado)
    return _features_de_normalizados(pts)


def _features_de_normalizados(pts: np.ndarray) -> np.ndarray:
    coords = pts.flatten()

    # distancias entre pontas dos dedos: separa letras parecidas (A/S/E, M/N/T)
    dist_pontas = []
    for i in range(len(PONTAS)):
        for j in range(i + 1, len(PONTAS)):
            dist_pontas.append(np.linalg.norm(pts[PONTAS[i]] - pts[PONTAS[j]]))

    # quanto cada dedo esta estendido
    dist_pulso = [np.linalg.norm(pts[p]) for p in PONTAS]

    return np.concatenate([coords, dist_pontas, dist_pulso]).astype(np.float32)


# --- Data augmentation ----------------------------------------------------
# So use no conjunto de TREINO. Nunca no de teste.

def _rot_z(graus: float) -> np.ndarray:
    r = np.deg2rad(graus)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _rot_y(graus: float) -> np.ndarray:
    r = np.deg2rad(graus)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def aumentar(pontos_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variacao plausivel de uma mesma pose (ja normalizada)."""
    pts = pontos_norm.copy()
    pts = pts @ _rot_z(rng.uniform(-12, 12)).T   # inclinacao no plano da imagem
    pts = pts @ _rot_y(rng.uniform(-10, 10)).T   # mao girando em relacao a camera
    pts += rng.normal(0, 0.012, pts.shape).astype(np.float32)  # ruido do detector
    return pts


def aumentar_lote(X_norm: np.ndarray, y: np.ndarray, copias: int = 3, seed: int = 42):
    """X_norm: (N, 21, 3) normalizados. Retorna features (N*(1+copias), DIM) e labels."""
    rng = np.random.default_rng(seed)
    feats = [_features_de_normalizados(p) for p in X_norm]
    labels = list(y)
    for _ in range(copias):
        for pts, rotulo in zip(X_norm, y):
            feats.append(_features_de_normalizados(aumentar(pts, rng)))
            labels.append(rotulo)
    return np.array(feats, dtype=np.float32), np.array(labels)

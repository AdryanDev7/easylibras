"""Fixtures compartilhadas: uma mao sintetica e um modelo pequeno treinado na hora."""
from __future__ import annotations

import numpy as np
import pytest


def mao_falsa(letra: str, rng: np.random.Generator) -> np.ndarray:
    """(21,3) em 0..1. Cada 'letra' e uma pose diferente + ruido."""
    base = np.zeros((21, 3), np.float32)
    semente = sum(map(ord, letra))
    forma = np.random.default_rng(semente).uniform(-0.15, 0.15, (21, 2))
    base[:, :2] = 0.5 + forma
    base[9, :2] = base[0, :2] + [0.0, -0.12]   # pulso -> medio_mcp com escala estavel
    base[:, :2] += rng.normal(0, 0.004, (21, 2))
    return base


@pytest.fixture(scope="session")
def modelo_treinado(tmp_path_factory):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from core_ai.features import VERSAO_FEATURES, montar_features

    rng = np.random.default_rng(0)
    X, y = [], []
    for letra in "ABC":
        for _ in range(40):
            pts = mao_falsa(letra, rng) * [1280, 720, 1280]
            X.append(montar_features(pts, "Right"))
            y.append(letra)
    m = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1).fit(X, y)
    caminho = tmp_path_factory.mktemp("modelo") / "libras_rf.joblib"
    joblib.dump({"modelo": m, "rotulos": ["A", "B", "C"], "versao_features": VERSAO_FEATURES},
                caminho)
    return caminho

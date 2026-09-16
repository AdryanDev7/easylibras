"""Carrega o modelo uma vez e faz a inferencia para a API."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from core_ai.config import CLASSIFIER_PATH
from core_ai.features import VERSAO_FEATURES, montar_features


class Predictor:
    def __init__(self, caminho: Path = CLASSIFIER_PATH):
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em {caminho}. Rode `python -m core_ai.train`."
            )
        pacote = joblib.load(caminho)
        if pacote.get("versao_features") != VERSAO_FEATURES:
            raise RuntimeError("Modelo treinado com outra versao de features. Retreine.")
        self.modelo = pacote["modelo"]
        self.rotulos = list(pacote["rotulos"])
        self.meta = {k: v for k, v in pacote.items() if k != "modelo"}

    def prever(self, pontos: np.ndarray, lado: str, top_k: int = 3) -> dict:
        """pontos: (21,3) em escala de pixels."""
        feats = montar_features(pontos, lado).reshape(1, -1)
        probs = self.modelo.predict_proba(feats)[0]
        ordem = np.argsort(probs)[::-1][:top_k]
        return {
            "letra": str(self.modelo.classes_[ordem[0]]),
            "confianca": float(probs[ordem[0]]),
            "top": [
                {"letra": str(self.modelo.classes_[i]), "confianca": float(probs[i])}
                for i in ordem
            ],
        }

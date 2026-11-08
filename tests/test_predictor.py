import numpy as np

from backend_api.services.predictor import Predictor
from tests.conftest import mao_falsa


def test_predictor_acerta_e_formata(modelo_treinado):
    p = Predictor(modelo_treinado)
    assert p.modelo.n_jobs == 1, "inferencia de 1 frame deve usar n_jobs=1"
    rng = np.random.default_rng(99)
    pts = mao_falsa("B", rng) * [1280, 720, 1280]
    r = p.prever(pts, "Right", top_k=2)
    assert r["letra"] == "B"
    assert len(r["top"]) == 2 and r["detectou_mao"] is True
    assert abs(sum(c["confianca"] for c in p.prever(pts, "Right", 99)["top"]) - 1) < 1e-6

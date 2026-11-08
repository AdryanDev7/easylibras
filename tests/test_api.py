"""Testes da API. Precisam de:  pip install httpx pytest"""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import mao_falsa  # noqa: E402


@pytest.fixture
def cliente(modelo_treinado, monkeypatch):
    import backend_api.app as api
    monkeypatch.setattr(api, "CLASSIFIER_PATH", modelo_treinado)
    # isola dos modelos reais em models/ (se voces ja tiverem treinado sinais)
    monkeypatch.setattr(api, "SINAIS_MODEL_PATH", modelo_treinado.parent / "nao_existe.joblib")
    # nao baixa o hand_landmarker nos testes
    import core_ai.hand_tracker as ht
    monkeypatch.setattr(ht, "garantir_modelo", lambda *a, **k: None)
    with TestClient(api.app) as c:
        yield c


def _payload(letra="A"):
    pts = mao_falsa(letra, np.random.default_rng(5))
    return {"landmarks": pts.tolist(), "lado": "Right", "img_w": 1280, "img_h": 720}


def test_health_e_labels(cliente):
    h = cliente.get("/health").json()
    assert h["modelo_carregado"] is True
    assert cliente.get("/labels").json()["letras"] == ["A", "B", "C"]


def test_predict(cliente):
    r = cliente.post("/predict?top_k=2", json=_payload("C"))
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["letra"] == "C" and len(corpo["top"]) == 2


def test_predict_valida(cliente):
    ruim = _payload(); ruim["landmarks"] = ruim["landmarks"][:5]
    assert cliente.post("/predict", json=ruim).status_code == 422


def test_frame_base64_invalido(cliente):
    r = cliente.post("/predict/frame", json={"imagem_base64": "!!!nao-e-base64"})
    assert r.status_code == 422


def test_websocket_sobrevive_a_lixo(cliente):
    with cliente.websocket_connect("/ws") as ws:
        ws.send_text("isso nao e json")
        assert "erro" in ws.receive_json()
        ws.send_text(json.dumps([1, 2, 3]))
        assert "erro" in ws.receive_json()
        ws.send_text(json.dumps({"landmarks": [[0, 0, 0]]}))
        assert "erro" in ws.receive_json()
        # e continua funcionando depois dos erros
        ws.send_text(json.dumps(_payload("B")))
        assert ws.receive_json()["letra"] == "B"


def test_reload(cliente):
    assert cliente.post("/reload").json()["ok"] is True


def test_sem_modelo_responde_503(cliente, monkeypatch, tmp_path):
    import backend_api.app as api
    monkeypatch.setattr(api, "CLASSIFIER_PATH", tmp_path / "nao_existe.joblib")
    assert cliente.post("/reload").status_code == 503
    assert cliente.post("/predict", json=_payload()).status_code == 503
    assert cliente.get("/health").json()["modelo_carregado"] is False


@pytest.fixture
def cliente_sinais(cliente, monkeypatch, tmp_path):
    import joblib
    import backend_api.app as api
    from core_ai.features_sinais import VERSAO_SINAIS, sequencia_para_matriz
    from core_ai.sinais import ClassificadorSinais
    from tests.sintetico_sinais import gerar
    M, S, G = [], [], []
    for s in ["OCULOS", "SAUDADE", "TCHAU"]:
        for p in range(3):
            M.append(sequencia_para_matriz(gerar(s, p, 0))); S.append(s); G.append(f"p{p}")
    caminho = tmp_path / "sinais.joblib"
    joblib.dump({"classificador": ClassificadorSinais().fit(M, S, G),
                 "versao_sinais": VERSAO_SINAIS}, caminho)
    monkeypatch.setattr(api, "SINAIS_MODEL_PATH", caminho)
    assert cliente.post("/reload").json()["sinais"] == ["OCULOS", "SAUDADE", "TCHAU"]
    return cliente


def test_sinais_predict(cliente_sinais):
    from tests.sintetico_sinais import H, W, gerar
    escala = np.array([W, H, W])
    quadros = [{"ts_ms": q.ts_ms,
                "maos": {k: (v / escala).tolist() for k, v in q.maos.items()},
                "pose": (q.pose / escala).tolist()} for q in gerar("SAUDADE", 8, 0)]
    r = cliente_sinais.post("/sinais/predict", json={"quadros": quadros, "img_w": W, "img_h": H})
    assert r.status_code == 200, r.text
    assert r.json()["sinal"] == "SAUDADE"
    assert cliente_sinais.get("/sinais/labels").json()["sinais"] == ["OCULOS", "SAUDADE", "TCHAU"]

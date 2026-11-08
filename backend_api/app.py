"""
API de traducao do EasyLibras.

    uvicorn backend_api.app:app --reload --host 0.0.0.0 --port 8000

Docs interativas: http://localhost:8000/docs

Dois caminhos possiveis:

1. POST /predict  (RECOMENDADO para o app mobile)
   O celular roda o MediaPipe localmente e manda so os 21 pontos.
   Payload de ~1 KB, latencia baixa, servidor barato.

2. POST /predict/frame
   O celular manda a imagem em base64 e o servidor roda o MediaPipe.
   Mais simples de implementar no app, porem bem mais pesado na rede e no
   servidor. Serve para prototipo/demo.
"""
from __future__ import annotations

import base64
import binascii
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend_api.schemas import FrameIn, LandmarksIn, PredicaoOut, SequenciaIn
from backend_api.services.predictor import Predictor
from core_ai.config import CLASSIFIER_PATH, HAND_MODEL_PATH, MAO_DOMINANTE, SINAIS_MODEL_PATH

VERSAO_API = "0.2.0"
WEB_DIR = Path(__file__).resolve().parent.parent / "web_demo"

predictor: Optional[Predictor] = None
erro_modelo: Optional[str] = None

# O HandLandmarker do MediaPipe NAO e thread-safe, e o FastAPI roda rotas `def`
# num pool de threads. Sem este lock, duas requisicoes simultaneas em
# /predict/frame podem corromper o estado do detector.
_tracker = None
_tracker_lock = threading.Lock()


classificador_sinais = None
erro_sinais: Optional[str] = None


def carregar_sinais() -> None:
    """Modelo de sinais dinamicos (opcional: a API sobe sem ele)."""
    global classificador_sinais, erro_sinais
    try:
        import joblib
        from core_ai.features_sinais import VERSAO_SINAIS
        if not SINAIS_MODEL_PATH.exists():
            raise FileNotFoundError(f"{SINAIS_MODEL_PATH.name} nao existe "
                                    "(rode python -m core_ai.treinar_sinais)")
        pacote = joblib.load(SINAIS_MODEL_PATH)
        if pacote.get("versao_sinais") != VERSAO_SINAIS:
            raise RuntimeError("modelo de sinais de outra versao de features; retreine")
        classificador_sinais, erro_sinais = pacote["classificador"], None
        print(f"[api] Sinais carregados: {classificador_sinais.sinais}")
    except Exception as exc:
        classificador_sinais, erro_sinais = None, str(exc)
        print(f"[api] Sinais dinamicos indisponiveis ({exc}).")


def carregar_modelo() -> None:
    global predictor, erro_modelo
    try:
        predictor = Predictor(CLASSIFIER_PATH)
        erro_modelo = None
        print(f"[api] Modelo carregado. Letras: {predictor.rotulos}")
    except Exception as exc:
        predictor = None
        erro_modelo = str(exc)
        print(f"[api] AVISO: modelo nao carregado ({exc}). Treine antes de usar /predict.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    carregar_modelo()
    carregar_sinais()

    # a demo web busca o hand_landmarker.task nesta mesma origem
    try:
        from core_ai.hand_tracker import garantir_modelo
        garantir_modelo()
    except Exception as exc:
        print(f"[api] AVISO: hand_landmarker.task indisponivel ({exc}). A demo web nao vai abrir.")

    if WEB_DIR.exists():
        print("[api] Demo web em http://localhost:8000/demo/")
    yield

    if _tracker is not None:
        _tracker.close()


app = FastAPI(title="EasyLibras API", version=VERSAO_API, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em producao, restrinja ao dominio do app
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/demo", StaticFiles(directory=WEB_DIR, html=True), name="demo")


@app.get("/", include_in_schema=False)
def raiz():
    return RedirectResponse("/demo/" if WEB_DIR.exists() else "/docs")


@app.get("/models/hand_landmarker.task", include_in_schema=False)
def modelo_mao():
    """Entrega o modelo do MediaPipe para o navegador, na mesma origem da API."""
    if not HAND_MODEL_PATH.exists():
        raise HTTPException(404, "hand_landmarker.task ainda nao foi baixado.")
    return FileResponse(HAND_MODEL_PATH, media_type="application/octet-stream")


def _exigir_modelo() -> Predictor:
    if predictor is None:
        raise HTTPException(503, f"Modelo nao carregado ({erro_modelo}). "
                                 "Rode `python -m core_ai.train` e chame POST /reload.")
    return predictor


def _para_pixels(dados: LandmarksIn) -> np.ndarray:
    """Formato ja validado pelo schema: (21,3) em 0..1 -> pixels."""
    pts = np.asarray(dados.landmarks, dtype=np.float32).copy()
    pts[:, 0] *= dados.img_w
    pts[:, 1] *= dados.img_h
    pts[:, 2] *= dados.img_w
    return pts


def _erros_legiveis(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


# --- Rotas ----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "versao_api": VERSAO_API,
        "modelo_carregado": predictor is not None,
        "erro_modelo": erro_modelo,
        "meta": predictor.meta if predictor else None,
        "sinais_carregados": classificador_sinais is not None,
        "erro_sinais": erro_sinais,
    }


@app.get("/labels")
def labels():
    return {"letras": _exigir_modelo().rotulos}


@app.post("/reload")
def reload():
    """Recarrega os modelos do disco (depois de retreinar, sem reiniciar o servidor)."""
    carregar_modelo()
    carregar_sinais()
    if predictor is None and classificador_sinais is None:
        raise HTTPException(503, f"Falhou: {erro_modelo} | {erro_sinais}")
    return {"ok": True,
            "letras": predictor.rotulos if predictor else None,
            "sinais": classificador_sinais.sinais if classificador_sinais else None,
            "meta": predictor.meta if predictor else None,
            "erro_modelo": erro_modelo, "erro_sinais": erro_sinais}


@app.post("/predict", response_model=PredicaoOut)
def predict(dados: LandmarksIn, top_k: int = Query(3, ge=1, le=30)):
    p = _exigir_modelo()
    return p.prever(_para_pixels(dados), dados.lado, top_k)


@app.post("/predict/frame", response_model=PredicaoOut)
def predict_frame(dados: FrameIn, top_k: int = Query(3, ge=1, le=30)):
    global _tracker
    p = _exigir_modelo()

    try:
        bruto = "".join(dados.imagem_base64.split(",")[-1].split())  # tira quebras de linha
        binario = base64.b64decode(bruto, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "base64 invalido")
    frame = cv2.imdecode(np.frombuffer(binario, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Nao consegui decodificar a imagem (mande JPEG ou PNG)")
    if dados.espelhar:
        frame = cv2.flip(frame, 1)

    with _tracker_lock:
        if _tracker is None:
            from core_ai.hand_tracker import HandTracker
            _tracker = HandTracker(num_maos=2, modo="imagem")
        maos = _tracker.processar(frame)

    from core_ai.hand_tracker import escolher_mao
    mao = escolher_mao(maos, MAO_DOMINANTE)
    if mao is None:
        return {"letra": None, "confianca": 0.0, "top": [], "detectou_mao": False}
    return p.prever(mao.pontos, mao.lado, top_k)


# --- Sinais dinamicos (palavras, sinais-nome) ------------------------------

@app.get("/sinais/labels")
def sinais_labels():
    if classificador_sinais is None:
        raise HTTPException(503, f"Modelo de sinais nao carregado ({erro_sinais}).")
    return {"sinais": classificador_sinais.sinais}


@app.post("/sinais/predict")
def sinais_predict(dados: SequenciaIn, top_k: int = Query(3, ge=1, le=30)):
    """Recebe a SEQUENCIA de frames de um sinal (ja cortada pelo cliente) e
    devolve o sinal reconhecido -- ou sinal=null se nao parecer nenhum conhecido."""
    if classificador_sinais is None:
        raise HTTPException(503, f"Modelo de sinais nao carregado ({erro_sinais}).")
    from core_ai.corpo_tracker import QuadroCorpo
    from core_ai.features_sinais import sequencia_para_matriz

    w, h = dados.img_w, dados.img_h
    escala = np.array([w, h, w], np.float32)
    quadros = [
        QuadroCorpo(
            maos={lado: np.asarray(pts, np.float32) * escala for lado, pts in q.maos.items()},
            pose=None if q.pose is None else np.asarray(q.pose, np.float32) * escala,
            img_w=w, img_h=h, ts_ms=q.ts_ms)
        for q in dados.quadros
    ]
    return classificador_sinais.prever(sequencia_para_matriz(quadros), top_k)


@app.websocket("/ws")
async def websocket_predict(ws: WebSocket):
    """Fluxo continuo: o cliente manda o mesmo JSON de /predict a cada frame.

    Toda mensagem recebe exatamente UMA resposta (a letra ou {"erro": ...}),
    e um erro nunca derruba a conexao. A demo web depende disso: ela so manda
    o proximo frame depois de receber a resposta do anterior.
    """
    await ws.accept()
    try:
        while True:
            texto = await ws.receive_text()
            try:
                payload = json.loads(texto)
                if not isinstance(payload, dict):
                    raise ValueError("esperado um objeto JSON")
                dados = LandmarksIn(**payload)
            except ValidationError as exc:
                await ws.send_json({"erro": _erros_legiveis(exc)})
                continue
            except ValueError as exc:  # inclui JSONDecodeError
                await ws.send_json({"erro": f"JSON invalido: {exc}"})
                continue

            p = predictor
            if p is None:
                await ws.send_json({"erro": "modelo nao carregado"})
                continue
            try:
                # a inferencia e CPU pura: rodar fora do event loop evita que um
                # cliente trave todos os outros
                resultado = await run_in_threadpool(p.prever, _para_pixels(dados), dados.lado)
            except Exception as exc:  # pragma: no cover
                resultado = {"erro": f"falha na inferencia: {exc}"}
            await ws.send_json(resultado)
    except WebSocketDisconnect:
        pass

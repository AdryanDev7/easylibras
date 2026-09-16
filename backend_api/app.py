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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend_api.services.predictor import Predictor
from core_ai.config import HAND_MODEL_PATH

WEB_DIR = Path(__file__).resolve().parent.parent / "web_demo"

predictor: Optional[Predictor] = None
tracker = None  # criado sob demanda, so se /predict/frame for usado


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor
    try:
        predictor = Predictor()
        print(f"[api] Modelo carregado. Letras: {predictor.rotulos}")
    except Exception as exc:
        print(f"[api] AVISO: modelo nao carregado ({exc}). Treine antes de usar /predict.")

    # a demo web busca o hand_landmarker.task nesta mesma origem
    try:
        from core_ai.hand_tracker import garantir_modelo
        garantir_modelo()
    except Exception as exc:
        print(f"[api] AVISO: hand_landmarker.task indisponivel ({exc}). A demo web nao vai abrir.")

    if WEB_DIR.exists():
        print(f"[api] Demo web em http://localhost:8000/demo/")
    yield


app = FastAPI(title="EasyLibras API", version="0.1.0", lifespan=lifespan)
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
        raise HTTPException(503, "Modelo nao carregado. Rode `python -m core_ai.train`.")
    return predictor


# --- Schemas --------------------------------------------------------------

class LandmarksIn(BaseModel):
    landmarks: List[List[float]] = Field(
        ..., description="21 pontos [x, y, z] normalizados 0..1 como o MediaPipe entrega"
    )
    lado: Literal["Right", "Left"] = "Right"
    img_w: int = 1280
    img_h: int = 720


class FrameIn(BaseModel):
    imagem_base64: str
    espelhar: bool = True


def _para_pixels(dados: LandmarksIn) -> np.ndarray:
    pts = np.asarray(dados.landmarks, dtype=np.float32)
    if pts.shape != (21, 3):
        raise HTTPException(422, f"Esperado shape (21,3), recebido {pts.shape}")
    pts = pts.copy()
    pts[:, 0] *= dados.img_w
    pts[:, 1] *= dados.img_h
    pts[:, 2] *= dados.img_w
    return pts


# --- Rotas ----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_carregado": predictor is not None,
        "meta": predictor.meta if predictor else None,
    }


@app.get("/labels")
def labels():
    return {"letras": _exigir_modelo().rotulos}


@app.post("/predict")
def predict(dados: LandmarksIn):
    p = _exigir_modelo()
    return p.prever(_para_pixels(dados), dados.lado)


@app.post("/predict/frame")
def predict_frame(dados: FrameIn):
    global tracker
    p = _exigir_modelo()

    from core_ai.hand_tracker import HandTracker
    if tracker is None:
        tracker = HandTracker(num_maos=1, modo="imagem")

    try:
        binario = base64.b64decode(dados.imagem_base64.split(",")[-1])
    except Exception:
        raise HTTPException(422, "base64 invalido")
    frame = cv2.imdecode(np.frombuffer(binario, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Nao consegui decodificar a imagem")
    if dados.espelhar:
        frame = cv2.flip(frame, 1)

    maos = tracker.processar(frame)
    if not maos:
        return {"letra": None, "confianca": 0.0, "detectou_mao": False}

    resultado = p.prever(maos[0].pontos, maos[0].lado)
    resultado["detectou_mao"] = True
    return resultado


@app.websocket("/ws")
async def websocket_predict(ws: WebSocket):
    """Fluxo continuo: o cliente manda o mesmo JSON de /predict a cada frame."""
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            if predictor is None:
                await ws.send_json({"erro": "modelo nao carregado"})
                continue
            try:
                dados = LandmarksIn(**payload)
                await ws.send_json(predictor.prever(_para_pixels(dados), dados.lado))
            except Exception as exc:
                await ws.send_json({"erro": str(exc)})
    except WebSocketDisconnect:
        pass

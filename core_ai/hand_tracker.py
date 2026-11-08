"""
Wrapper do MediaPipe Hand Landmarker.

IMPORTANTE: usa a API NOVA (mediapipe.tasks). O modulo antigo
`mp.solutions.hands` foi REMOVIDO das versoes recentes do mediapipe,
entao todo tutorial que usa `mp.solutions.hands` quebra hoje.
"""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

from .config import HAND_MODEL_PATH, HAND_MODEL_URL

CONEXOES = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]


@dataclass
class MaoDetectada:
    """Uma mao detectada em um frame."""
    pontos: np.ndarray   # (21, 3) em escala de PIXELS (x,y) + z relativo
    lado: str            # "Right" ou "Left" conforme o MediaPipe
    score: float


def garantir_modelo(destino: Path = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> Path:
    """Baixa o modelo (.task) do MediaPipe se ainda nao existir."""
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 1000:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hand_tracker] Baixando modelo do MediaPipe -> {destino}")
    try:
        urllib.request.urlretrieve(url, destino)
    except Exception as exc:  # pragma: no cover
        print(
            f"\n[ERRO] Nao consegui baixar o modelo automaticamente ({exc}).\n"
            f"Baixe manualmente:\n  {url}\n"
            f"e salve em:\n  {destino}\n",
            file=sys.stderr,
        )
        raise
    print("[hand_tracker] Modelo baixado.")
    return destino


class HandTracker:
    """Detector de mao pronto para loop de webcam.

    modo="video"  -> usa tracking entre frames (webcam). Exige timestamp crescente.
    modo="imagem" -> deteccao independente por frame (util para processar arquivos).
    """

    def __init__(self, num_maos: int = 1, modo: str = "video", conf: float = 0.5):
        caminho = garantir_modelo()
        running = (
            vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE
        )
        opcoes = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho)),
            running_mode=running,
            num_hands=num_maos,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opcoes)
        self._modo = modo

    def processar(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[MaoDetectada]:
        altura, largura = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        if self._modo == "video":
            resultado = self._landmarker.detect_for_video(imagem, int(timestamp_ms))
        else:
            resultado = self._landmarker.detect(imagem)

        maos: list[MaoDetectada] = []
        for i, marcos in enumerate(resultado.hand_landmarks):
            pontos = np.array(
                [[lm.x * largura, lm.y * altura, lm.z * largura] for lm in marcos],
                dtype=np.float32,
            )
            cat = resultado.handedness[i][0]
            maos.append(MaoDetectada(pontos=pontos, lado=cat.category_name, score=cat.score))
        return maos

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def escolher_mao(maos: list[MaoDetectada], dominante: str = "Right") -> MaoDetectada | None:
    """Com 2 maos na tela, devolve a dominante; se ela nao aparece, a de maior score."""
    if not maos:
        return None
    for m in maos:
        if m.lado == dominante:
            return m
    return max(maos, key=lambda m: m.score)


def desenhar_mao(frame: np.ndarray, pontos: np.ndarray, cor=(0, 220, 120)) -> None:
    """Desenha esqueleto da mao no frame (in-place).

    O mediapipe novo nao traz mais `drawing_utils`, entao desenhamos na mao.
    """
    pts = pontos[:, :2].astype(int)
    for a, b in CONEXOES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), cor, 2, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 4, cor, 1, cv2.LINE_AA)

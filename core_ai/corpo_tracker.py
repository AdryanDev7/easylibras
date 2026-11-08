"""
Rastreador para sinais dinamicos: DUAS maos + pose (rosto e ombros).

Por que a pose? Em Libras o MESMO formato de mao significa coisas diferentes
dependendo de ONDE ela esta: perto do olho (OCULOS), na bochecha (covinha),
no peito (SAUDADE). Sem saber onde estao o rosto e os ombros, o modelo so ve
a mao "flutuando" e nao tem como distinguir esses sinais.

Usamos dois modelos do MediaPipe (API nova, mediapipe.tasks):
  - HandLandmarker com num_hands=2
  - PoseLandmarker (versao lite): 33 pontos, dos quais usamos rosto e ombros
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .config import POSE_MODEL_PATH, POSE_MODEL_URL
from .hand_tracker import HandTracker, desenhar_mao, garantir_modelo

# Indices da pose do MediaPipe que interessam (rosto + ombros)
NARIZ, OLHO_E, OLHO_D, ORELHA_E, ORELHA_D, BOCA_E, BOCA_D, OMBRO_E, OMBRO_D = \
    0, 2, 5, 7, 8, 9, 10, 11, 12


@dataclass
class QuadroCorpo:
    """Tudo o que foi detectado em UM frame, em pixels."""
    maos: dict = field(default_factory=dict)   # {"Right": (21,3), "Left": (21,3)}
    pose: np.ndarray | None = None             # (33,3) ou None
    img_w: int = 0
    img_h: int = 0
    ts_ms: int = 0


class CorpoTracker:
    def __init__(self, conf: float = 0.5):
        self._maos = HandTracker(num_maos=2, modo="video", conf=conf)
        caminho = garantir_modelo(POSE_MODEL_PATH, POSE_MODEL_URL)
        opcoes = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=conf,
            min_pose_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._pose = vision.PoseLandmarker.create_from_options(opcoes)

    def processar(self, frame_bgr: np.ndarray, ts_ms: int) -> QuadroCorpo:
        h, w = frame_bgr.shape[:2]
        quadro = QuadroCorpo(img_w=w, img_h=h, ts_ms=int(ts_ms))

        # Se o MediaPipe chamar as duas maos de "Right" (acontece), a de maior
        # score fica com o rotulo e a outra vai para o lado livre.
        maos = sorted(self._maos.processar(frame_bgr, ts_ms), key=lambda m: -m.score)
        for m in maos:
            lado = m.lado if m.lado not in quadro.maos else \
                ("Left" if m.lado == "Right" else "Right")
            if lado not in quadro.maos:
                quadro.maos[lado] = m.pontos

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._pose.detect_for_video(imagem, int(ts_ms))
        if res.pose_landmarks:
            quadro.pose = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in res.pose_landmarks[0]],
                dtype=np.float32,
            )
        return quadro

    def close(self):
        self._maos.close()
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def desenhar_quadro(frame: np.ndarray, quadro: QuadroCorpo, referencias: dict | None = None):
    """Desenha maos (direita verde, esquerda azul), ombros e os pontos de referencia."""
    if quadro.pose is not None:
        p = quadro.pose
        for a, b in [(OMBRO_E, OMBRO_D)]:
            cv2.line(frame, tuple(p[a, :2].astype(int)), tuple(p[b, :2].astype(int)),
                     (90, 90, 90), 2, cv2.LINE_AA)
    if referencias:
        for nome, (x, y) in referencias.items():
            cv2.circle(frame, (int(x), int(y)), 5, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, nome, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 200, 255), 1, cv2.LINE_AA)
    cores = {"Right": (0, 220, 120), "Left": (255, 160, 0)}
    for lado, pts in quadro.maos.items():
        desenhar_mao(frame, pts, cores.get(lado, (200, 200, 200)))


# --------------------------------------------------------------------------
# Formato BRUTO de uma gravacao (o que vai para o disco)
# --------------------------------------------------------------------------
# Guardamos os pontos crus em pixels; as features sao calculadas depois.
# Assim da para mudar features_sinais.py sem regravar nada.

def quadros_para_bruto(quadros: list[QuadroCorpo]) -> dict:
    n = len(quadros)
    direita = np.full((n, 21, 3), np.nan, np.float32)
    esquerda = np.full((n, 21, 3), np.nan, np.float32)
    pose = np.full((n, 33, 3), np.nan, np.float32)
    for i, q in enumerate(quadros):
        if "Right" in q.maos:
            direita[i] = q.maos["Right"]
        if "Left" in q.maos:
            esquerda[i] = q.maos["Left"]
        if q.pose is not None:
            pose[i] = q.pose
    return {
        "direita": direita, "esquerda": esquerda, "pose": pose,
        "ts": np.array([q.ts_ms for q in quadros], np.int64),
        "img_w": quadros[0].img_w if quadros else 0,
        "img_h": quadros[0].img_h if quadros else 0,
    }


def bruto_para_quadros(bruto: dict) -> list[QuadroCorpo]:
    quadros = []
    for i in range(len(bruto["ts"])):
        maos = {}
        if np.isfinite(bruto["direita"][i]).all():
            maos["Right"] = bruto["direita"][i]
        if np.isfinite(bruto["esquerda"][i]).all():
            maos["Left"] = bruto["esquerda"][i]
        pose = bruto["pose"][i] if np.isfinite(bruto["pose"][i]).all() else None
        quadros.append(QuadroCorpo(maos=maos, pose=pose, img_w=int(bruto["img_w"]),
                                   img_h=int(bruto["img_h"]), ts_ms=int(bruto["ts"][i])))
    return quadros

"""Contratos de entrada/saida da API (so depende de pydantic, facil de testar)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ~4 MB de imagem em base64 ja e muito mais do que uma foto 1280x720 em JPEG.
MAX_BASE64_CHARS = 6_000_000


def normalizar_lado(v) -> str:
    t = str(v).strip().lower()
    if t in ("right", "r", "direita", "d"):
        return "Right"
    if t in ("left", "l", "esquerda", "e"):
        return "Left"
    raise ValueError('lado deve ser "Right" ou "Left"')


class LandmarksIn(BaseModel):
    landmarks: List[List[float]] = Field(
        ...,
        description="21 pontos [x, y, z] normalizados 0..1, exatamente como o MediaPipe entrega",
        examples=[[[0.5, 0.5, 0.0]] * 21],
    )
    lado: str = Field("Right", description='"Right" ou "Left" (o handedness do MediaPipe)')
    img_w: int = Field(1280, gt=0, le=10000, description="largura da imagem em pixels")
    img_h: int = Field(720, gt=0, le=10000, description="altura da imagem em pixels")

    @field_validator("landmarks")
    @classmethod
    def _formato(cls, v):
        if len(v) != 21:
            raise ValueError(f"esperado 21 pontos, recebido {len(v)}")
        for i, p in enumerate(v):
            if len(p) not in (2, 3):
                raise ValueError(f"ponto {i}: esperado [x, y, z], recebido {len(p)} valores")
            if not all(math.isfinite(c) for c in p):
                raise ValueError(f"ponto {i}: contem NaN/infinito")
        # aceita [x, y] sem z
        return [list(p) + [0.0] if len(p) == 2 else list(p) for p in v]

    @field_validator("lado")
    @classmethod
    def _lado(cls, v):
        return normalizar_lado(v)


class FrameIn(BaseModel):
    imagem_base64: str = Field(..., max_length=MAX_BASE64_CHARS,
                               description="JPEG/PNG em base64 (aceita prefixo data:image/...;base64,)")
    espelhar: bool = Field(True, description="aplica cv2.flip como na coleta. "
                                             "Mande False se o app ja espelhou a imagem.")


class Candidato(BaseModel):
    letra: str
    confianca: float


class PredicaoOut(BaseModel):
    letra: Optional[str]
    confianca: float
    top: List[Candidato] = []
    detectou_mao: bool = True


# --- Sinais dinamicos ------------------------------------------------------

MAX_QUADROS = 300   # ~10 s a 30 fps


def _validar_pontos(pts, n: int, nome: str):
    if len(pts) != n:
        raise ValueError(f"{nome}: esperado {n} pontos, recebido {len(pts)}")
    for i, p in enumerate(pts):
        if len(p) not in (2, 3) or not all(math.isfinite(c) for c in p):
            raise ValueError(f"{nome}, ponto {i}: esperado [x, y, z] finitos")
    return [list(p) + [0.0] if len(p) == 2 else list(p) for p in pts]


class QuadroIn(BaseModel):
    """Um frame: maos e pose normalizadas 0..1, como o MediaPipe entrega."""
    ts_ms: int = Field(..., ge=0, description="timestamp do frame em ms")
    maos: Dict[str, List[List[float]]] = Field(
        default_factory=dict, description='{"Right": 21 pontos, "Left": 21 pontos} (so as visiveis)')
    pose: Optional[List[List[float]]] = Field(None, description="33 pontos do PoseLandmarker")

    @field_validator("maos")
    @classmethod
    def _maos(cls, v):
        saida = {}
        for lado, pts in v.items():
            lado_ok = normalizar_lado(lado)
            saida[lado_ok] = _validar_pontos(pts, 21, f"mao {lado}")
        return saida

    @field_validator("pose")
    @classmethod
    def _pose(cls, v):
        return None if v is None else _validar_pontos(v, 33, "pose")


class SequenciaIn(BaseModel):
    quadros: List[QuadroIn] = Field(..., min_length=3, max_length=MAX_QUADROS)
    img_w: int = Field(1280, gt=0, le=10000)
    img_h: int = Field(720, gt=0, le=10000)

import math

import pytest
from pydantic import ValidationError

from backend_api.schemas import FrameIn, LandmarksIn, MAX_BASE64_CHARS

PONTOS = [[0.5, 0.5, 0.0]] * 21


def test_aceita_payload_da_demo_web():
    d = LandmarksIn(landmarks=PONTOS, lado="Left", img_w=640, img_h=480)
    assert d.lado == "Left" and len(d.landmarks) == 21


def test_lado_tolerante():
    assert LandmarksIn(landmarks=PONTOS, lado="right").lado == "Right"
    assert LandmarksIn(landmarks=PONTOS, lado="esquerda").lado == "Left"
    with pytest.raises(ValidationError):
        LandmarksIn(landmarks=PONTOS, lado="meio")


def test_aceita_xy_sem_z():
    d = LandmarksIn(landmarks=[[0.1, 0.2]] * 21)
    assert d.landmarks[0] == [0.1, 0.2, 0.0]


@pytest.mark.parametrize("pts", [
    [[0.5, 0.5, 0.0]] * 20,                      # faltando ponto
    [[0.5, 0.5, 0.0, 1.0]] * 21,                 # 4 coordenadas
    [[math.nan, 0.5, 0.0]] + [[0.5, 0.5, 0.0]] * 20,
    [[math.inf, 0.5, 0.0]] + [[0.5, 0.5, 0.0]] * 20,
])
def test_rejeita_landmarks_invalidos(pts):
    with pytest.raises(ValidationError):
        LandmarksIn(landmarks=pts)


@pytest.mark.parametrize("w,h", [(0, 720), (1280, 0), (-1, 720)])
def test_rejeita_resolucao_invalida(w, h):
    with pytest.raises(ValidationError):
        LandmarksIn(landmarks=PONTOS, img_w=w, img_h=h)


def test_limita_tamanho_da_imagem():
    with pytest.raises(ValidationError):
        FrameIn(imagem_base64="A" * (MAX_BASE64_CHARS + 1))


def test_sequencia_de_sinal():
    from backend_api.schemas import SequenciaIn
    q = {"ts_ms": 0, "maos": {"right": [[0.5, 0.5]] * 21}, "pose": [[0.5, 0.5, 0]] * 33}
    s = SequenciaIn(quadros=[q] * 3)
    assert list(s.quadros[0].maos) == ["Right"] and len(s.quadros[0].maos["Right"][0]) == 3
    with pytest.raises(ValidationError):
        SequenciaIn(quadros=[q] * 2)                         # curta demais
    with pytest.raises(ValidationError):
        SequenciaIn(quadros=[{**q, "pose": [[0, 0, 0]] * 10}] * 3)

"""
Features dos sinais DINAMICOS (palavras e sinais-nome).

Um sinal de Libras tem cinco parametros: configuracao de mao, ponto de
articulacao (ONDE), movimento, orientacao da palma e expressao nao manual.
Cada frame vira um vetor que cobre os quatro primeiros:

  por mao (dominante e nao dominante):
    presente            a mao esta na tela?
    posicao  (6)        pulso, ponta do indicador e do polegar, no referencial do CORPO
    forma    (28)       pontas dos dedos normalizadas, distancias entre pontas, normal da palma
    contato  (9)        distancia ate olho, nariz, boca, bochecha, queixo, testa,
                        cabelo, orelha e peito
  entre as maos (2)     distancia entre pulsos e entre pontas dos indicadores

O MOVIMENTO nao vira feature: ele aparece como a SEQUENCIA de frames, que e o
que o DTW (core_ai/sinais.py) compara.

Referencial do corpo: origem no meio dos ombros, unidade = largura dos ombros,
y para baixo. Assim o sinal fica igual com a pessoa perto ou longe da camera.

Expressao facial (ex.: bochecha inflada) fica de fora nesta versao -- e um bom
"trabalho futuro" no relatorio (FaceLandmarker do MediaPipe).

Mudou alguma coisa aqui? Suba VERSAO_SINAIS e rode o treino de novo.
"""
from __future__ import annotations

import numpy as np

from .config import MAO_DOMINANTE, SINAIS_FPS
from .corpo_tracker import (BOCA_D, BOCA_E, NARIZ, OLHO_D, OLHO_E, OMBRO_D, OMBRO_E,
                            ORELHA_D, ORELHA_E, QuadroCorpo)
from .features import PONTAS, normalizar_pontos

VERSAO_SINAIS = 1

REFERENCIAS = ["olho", "nariz", "boca", "bochecha", "queixo", "testa", "cabelo",
               "orelha", "peito"]
DIST_MAX = 1.5   # distancias maiores que isso (em larguras de ombro) viram "longe"

# Grupos de colunas e o peso de cada um na comparacao. Cada grupo e dividido
# pela raiz do seu tamanho, para que 28 colunas de "forma" nao abafem as 9 de
# "contato" -- e o contato que separa OCULOS (olho) de covinha (bochecha).
_GRUPOS_MAO = [("presente", 1), ("posicao", 6), ("forma", 28), ("contato", len(REFERENCIAS))]
PESOS = {"presente": 1.0, "posicao": 1.5, "forma": 1.0, "contato": 1.5, "entre": 0.7}
DIM_MAO = sum(n for _, n in _GRUPOS_MAO)
DIM_SINAL = 2 * DIM_MAO + 2


def grupos_colunas() -> list[str]:
    """Nome do grupo de cada coluna do vetor (usado para os pesos)."""
    nomes = []
    for _ in range(2):
        for g, n in _GRUPOS_MAO:
            nomes += [g] * n
    return nomes + ["entre"] * 2


# --------------------------------------------------------------------------
# Referencial do corpo
# --------------------------------------------------------------------------

class Referencial:
    """Origem/escala do corpo + pontos do rosto, em pixels."""

    def __init__(self, pose: np.ndarray):
        p = pose[:, :2]
        self.origem = (p[OMBRO_E] + p[OMBRO_D]) / 2
        self.escala = max(float(np.linalg.norm(p[OMBRO_E] - p[OMBRO_D])), 1.0)
        olhos = (p[OLHO_E] + p[OLHO_D]) / 2
        boca = (p[BOCA_E] + p[BOCA_D]) / 2
        nariz = p[NARIZ]
        acima = olhos - boca                 # direcao "para cima" do rosto
        # Pose nao tem bochecha/queixo/testa: aproximamos a partir dos pontos que tem.
        # Varios pontos por referencia = "qualquer um dos dois lados".
        self.pontos = {
            "olho": [p[OLHO_E], p[OLHO_D]],
            "nariz": [nariz],
            "boca": [boca],
            "bochecha": [(p[BOCA_E] + p[ORELHA_E]) / 2, (p[BOCA_D] + p[ORELHA_D]) / 2],
            "queixo": [boca - 0.9 * acima],
            "testa": [olhos + 0.9 * acima],
            "cabelo": [olhos + 2.0 * acima, p[ORELHA_E] + 1.2 * acima, p[ORELHA_D] + 1.2 * acima],
            "orelha": [p[ORELHA_E], p[ORELHA_D]],
            "peito": [self.origem + np.array([0.0, 0.45 * self.escala])],
        }

    def para_corpo(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, np.float32)[..., :2] - self.origem) / self.escala

    def pontos_desenho(self) -> dict:
        return {nome: pts[0] for nome, pts in self.pontos.items()}


def referencial_padrao(img_w: int, img_h: int) -> "Referencial":
    """Usado so se a pose nunca aparecer: um corpo 'medio' centrado na imagem."""
    pose = np.zeros((33, 3), np.float32)
    cx, ombro_y, larg = img_w / 2, img_h * 0.85, img_w * 0.35
    pose[OMBRO_E, :2] = [cx + larg / 2, ombro_y]
    pose[OMBRO_D, :2] = [cx - larg / 2, ombro_y]
    face_y = ombro_y - larg * 0.9
    pose[NARIZ, :2] = [cx, face_y]
    pose[OLHO_E, :2] = [cx + larg * 0.08, face_y - larg * 0.08]
    pose[OLHO_D, :2] = [cx - larg * 0.08, face_y - larg * 0.08]
    pose[BOCA_E, :2] = [cx + larg * 0.06, face_y + larg * 0.1]
    pose[BOCA_D, :2] = [cx - larg * 0.06, face_y + larg * 0.1]
    pose[ORELHA_E, :2] = [cx + larg * 0.2, face_y]
    pose[ORELHA_D, :2] = [cx - larg * 0.2, face_y]
    return Referencial(pose)


# --------------------------------------------------------------------------
# Vetor de UM frame
# --------------------------------------------------------------------------

def _forma(pts_px: np.ndarray, lado: str) -> np.ndarray:
    n = normalizar_pontos(pts_px, lado)          # mesma normalizacao do alfabeto
    pontas = n[PONTAS].ravel()                   # 15
    dists = [np.linalg.norm(n[PONTAS[i]] - n[PONTAS[j]])
             for i in range(5) for j in range(i + 1, 5)]   # 10
    normal = np.cross(n[5] - n[0], n[17] - n[0])  # palma: indicador x minimo
    normal = normal / (np.linalg.norm(normal) + 1e-6)       # 3
    return np.concatenate([pontas, dists, normal])


def _vetor_mao(pts_px: np.ndarray | None, lado: str, ref: Referencial) -> np.ndarray:
    v = np.zeros(DIM_MAO, np.float32)
    if pts_px is None:
        v[-len(REFERENCIAS):] = DIST_MAX         # ausente = longe de tudo
        return v
    corpo = ref.para_corpo(pts_px)               # (21,2)
    v[0] = 1.0
    v[1:7] = np.concatenate([corpo[0], corpo[8], corpo[4]])
    v[7:35] = _forma(pts_px, lado)
    # contato: a menor distancia entre (polegar, indicador, medio, palma) e cada referencia
    palma = (corpo[0] + corpo[9]) / 2
    toque = np.vstack([corpo[4], corpo[8], corpo[12], palma])
    for k, nome in enumerate(REFERENCIAS):
        alvo = ref.para_corpo(np.array(ref.pontos[nome]))
        d = np.linalg.norm(toque[:, None, :] - alvo[None, :, :], axis=-1).min()
        v[35 + k] = min(float(d), DIST_MAX)
    return v


def vetor_quadro(quadro: QuadroCorpo, ref: Referencial, dominante: str = MAO_DOMINANTE) -> np.ndarray:
    outra = "Left" if dominante == "Right" else "Right"
    pd, po = quadro.maos.get(dominante), quadro.maos.get(outra)
    v_dom = _vetor_mao(pd, dominante, ref)
    v_out = _vetor_mao(po, outra, ref)
    entre = np.array([DIST_MAX, DIST_MAX], np.float32)
    if pd is not None and po is not None:
        a, b = ref.para_corpo(pd), ref.para_corpo(po)
        entre = np.minimum([np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[8] - b[8])], DIST_MAX)
    return np.concatenate([v_dom, v_out, entre]).astype(np.float32)


# --------------------------------------------------------------------------
# Sequencia inteira
# --------------------------------------------------------------------------

def aparar(quadros: list[QuadroCorpo]) -> list[QuadroCorpo]:
    """Tira os frames sem mao do comeco e do fim."""
    com_mao = [i for i, q in enumerate(quadros) if q.maos]
    if not com_mao:
        return []
    return quadros[com_mao[0]:com_mao[-1] + 1]


def reamostrar(quadros: list[QuadroCorpo], fps: int = SINAIS_FPS) -> list[QuadroCorpo]:
    """Deixa a sequencia com fps fixo (a webcam de cada um roda num fps diferente)."""
    if len(quadros) < 2:
        return list(quadros)
    ts = np.array([q.ts_ms for q in quadros], np.float64)
    passo = 1000.0 / fps
    alvo = np.arange(ts[0], ts[-1] + 1e-6, passo)
    idx = np.clip(np.searchsorted(ts, alvo), 0, len(ts) - 1)
    # pega o vizinho mais proximo (anterior ou seguinte)
    anterior = np.clip(idx - 1, 0, len(ts) - 1)
    usar_ant = np.abs(ts[anterior] - alvo) < np.abs(ts[idx] - alvo)
    idx = np.where(usar_ant, anterior, idx)
    return [quadros[i] for i in idx]


def sequencia_para_matriz(quadros: list[QuadroCorpo], dominante: str = MAO_DOMINANTE,
                          fps: int = SINAIS_FPS) -> np.ndarray:
    """Lista de quadros (pixels) -> matriz (T, DIM_SINAL). Vazia se nao houver mao."""
    quadros = reamostrar(aparar(quadros), fps)
    if not quadros:
        return np.zeros((0, DIM_SINAL), np.float32)

    # pose ausente num frame: usa a ultima vista (o corpo quase nao se mexe)
    primeira = next((q.pose for q in quadros if q.pose is not None), None)
    ref = Referencial(primeira) if primeira is not None else \
        referencial_padrao(quadros[0].img_w, quadros[0].img_h)
    linhas = []
    for q in quadros:
        if q.pose is not None:
            ref = Referencial(q.pose)
        linhas.append(vetor_quadro(q, ref, dominante))
    return np.stack(linhas)


def espelhar_quadros(quadros: list[QuadroCorpo]) -> list[QuadroCorpo]:
    """Versao espelhada do sinal, como se feito por um canhoto.

    Inverte x de tudo e troca as maos de lado. Usado como data augmentation:
    com isso um sinal gravado por destro tambem reconhece o canhoto.
    """
    pares = [(OLHO_E, OLHO_D), (ORELHA_E, ORELHA_D), (BOCA_E, BOCA_D), (OMBRO_E, OMBRO_D),
             (1, 4), (3, 6), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
             (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    saida = []
    for q in quadros:
        w = q.img_w
        maos = {}
        for lado, pts in q.maos.items():
            p = pts.copy()
            p[:, 0] = w - p[:, 0]
            maos["Left" if lado == "Right" else "Right"] = p
        pose = None
        if q.pose is not None:
            pose = q.pose.copy()
            pose[:, 0] = w - pose[:, 0]
            for a, b in pares:
                pose[[a, b]] = pose[[b, a]]
        saida.append(QuadroCorpo(maos=maos, pose=pose, img_w=q.img_w, img_h=q.img_h,
                                 ts_ms=q.ts_ms))
    return saida

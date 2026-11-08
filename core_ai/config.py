"""Configuracoes centrais do EasyLibras."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"

RAW_CSV = DATASET_DIR / "raw_landmarks.csv"
PROCESSED_NPZ = DATASET_DIR / "processed.npz"
CLASSIFIER_PATH = MODEL_DIR / "libras_rf.joblib"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# Modelo oficial do MediaPipe Hand Landmarker (baixado automaticamente na 1a execucao)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Modelo de POSE (rosto + ombros), usado nos sinais dinamicos: e ele que diz
# se a mao esta no olho, na bochecha, no peito... (baixado na 1a execucao)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# --- Sinais dinamicos (palavras, sinais-nome) -----------------------------
SINAIS_DIR = DATASET_DIR / "sinais"          # uma pasta por sinal, um .npz por gravacao
SINAIS_MODEL_PATH = MODEL_DIR / "sinais_knn.joblib"
SINAIS_FPS = 15               # as gravacoes sao reamostradas para este fps
SINAIS_MAX_SEGUNDOS = 5.0     # duracao maxima de uma gravacao de sinal

# --- Alfabeto -------------------------------------------------------------
# Letras que sao POSE estatica: da para classificar com 1 frame. Comece por aqui.
LETRAS_ESTATICAS = list("ABCDEFGILMNOPQRSTUVWY")

# Letras que envolvem MOVIMENTO: precisam de modelo temporal (ver GUIA.md).
# Nao tente treinar essas com frame unico, o modelo vai confundir.
LETRAS_DINAMICAS = ["H", "J", "K", "X", "Z", "C_CEDILHA"]

# Quantas amostras coletar por letra, por sessao.
AMOSTRAS_PADRAO = 300

# Mao que faz as letras. Com duas maos na tela, o alfabeto usa esta.
# "Right" para destros, "Left" para canhotos.
MAO_DOMINANTE = "Right"

# --- Inferencia em tempo real --------------------------------------------
CONF_MINIMA = 0.65        # abaixo disso a predicao e descartada
JANELA_SUAVIZACAO = 8     # frames considerados na votacao
FRAMES_PARA_CONFIRMAR = 6 # frames estaveis para "escrever" a letra
SEGUNDOS_PARA_ESPACO = 1.2

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

"""
Tradutor ao vivo pela webcam.

    python -m core_ai.realtime

Teclas: ESC sai | C limpa o texto | B apaga a ultima letra

A predicao bruta de cada frame pisca muito. Por isso usamos uma janela de
votacao: a letra so e "escrita" quando aparece de forma estavel por varios
frames seguidos e com confianca acima do limiar.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import (CLASSIFIER_PATH, CONF_MINIMA, FRAMES_PARA_CONFIRMAR,
                     JANELA_SUAVIZACAO, MAO_DOMINANTE, SEGUNDOS_PARA_ESPACO)
from .features import VERSAO_FEATURES, montar_features
from .hand_tracker import HandTracker, desenhar_mao, escolher_mao


def carregar_modelo(caminho: Path):
    if not caminho.exists():
        raise SystemExit(f"Nao achei o modelo em {caminho}. Rode: python -m core_ai.train")
    pacote = joblib.load(caminho)
    if pacote.get("versao_features") != VERSAO_FEATURES:
        raise SystemExit(
            "Modelo treinado com outra versao de features. Retreine "
            "(preprocess + train) antes de usar."
        )
    modelo = pacote["modelo"]
    # n_jobs=-1 e otimo no treino, mas para prever 1 frame abre um pool de
    # threads a cada chamada (medido ~4x mais lento). Em tempo real, use 1.
    if hasattr(modelo, "n_jobs"):
        modelo.n_jobs = 1
    return modelo, pacote["rotulos"]


def _texto(frame, txt, pos, cor=(255, 255, 255), escala=0.7, grossura=1):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grossura + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def rodar(caminho_modelo: Path, camera: int, conf_min: float, resolucao: str = "1280x720"):
    modelo, rotulos = carregar_modelo(caminho_modelo)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera {camera}.")
    larg, alt = (int(v) for v in resolucao.lower().split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, larg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alt)

    tracker = HandTracker(num_maos=2, modo="video")
    historico = deque(maxlen=JANELA_SUAVIZACAO)
    texto = ""
    ultima_escrita = None
    estaveis = 0
    ultimo_visto = time.time()
    t0 = time.time()
    fps_t, fps = time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # MESMO espelhamento usado na coleta
            h, w = frame.shape[:2]
            maos = tracker.processar(frame, int((time.time() - t0) * 1000))

            letra, prob = None, 0.0

            if maos:
                mao = escolher_mao(maos, MAO_DOMINANTE)
                for outra in maos:
                    if outra is not mao:
                        desenhar_mao(frame, outra.pontos, (150, 150, 150))
                desenhar_mao(frame, mao.pontos)
                feats = montar_features(mao.pontos, mao.lado).reshape(1, -1)
                probs = modelo.predict_proba(feats)[0]
                k = int(np.argmax(probs))
                letra, prob = modelo.classes_[k], float(probs[k])

                historico.append(letra if prob >= conf_min else None)
                ultimo_visto = time.time()

                votos = [v for v in historico if v]
                if votos:
                    vencedora, n = Counter(votos).most_common(1)[0]
                    if n >= JANELA_SUAVIZACAO // 2 and vencedora == letra:
                        estaveis += 1
                    else:
                        estaveis = 0
                    if estaveis >= FRAMES_PARA_CONFIRMAR and vencedora != ultima_escrita:
                        texto += vencedora
                        ultima_escrita = vencedora
                        estaveis = 0
            else:
                historico.clear()
                estaveis = 0
                ultima_escrita = None  # libera repetir a mesma letra (ex.: "SS")
                if texto and not texto.endswith(" ") and \
                        time.time() - ultimo_visto > SEGUNDOS_PARA_ESPACO:
                    texto += " "

            # --- HUD ---
            cv2.rectangle(frame, (0, 0), (w, 92), (20, 20, 20), -1)
            if letra:
                cor = (0, 220, 120) if prob >= conf_min else (60, 160, 255)
                _texto(frame, f"{letra}", (20, 70), cor, 2.0, 3)
                barra = int(prob * 260)
                cv2.rectangle(frame, (110, 40), (110 + 260, 62), (70, 70, 70), -1)
                cv2.rectangle(frame, (110, 40), (110 + barra, 62), cor, -1)
                _texto(frame, f"{prob * 100:4.1f}%", (390, 60), cor, 0.7)
            else:
                _texto(frame, "mostre a mao", (20, 60), (120, 120, 120), 0.9)

            _texto(frame, f"{fps:4.1f} fps", (w - 130, 30), (150, 150, 150), 0.6)

            cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
            _texto(frame, texto[-40:] or "_", (20, h - 32), (255, 255, 255), 1.1, 2)
            _texto(frame, "ESC sair | C limpar | B apagar", (w - 340, h - 8), (140, 140, 140), 0.5)

            cv2.imshow("EasyLibras - tempo real", frame)

            agora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(agora - fps_t, 1e-6))
            fps_t = agora

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            if tecla in (ord("c"), ord("C")):
                texto, ultima_escrita = "", None
            if tecla in (ord("b"), ord("B")):
                texto = texto[:-1]
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTexto final: {texto!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", type=Path, default=CLASSIFIER_PATH)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=CONF_MINIMA)
    p.add_argument("--resolucao", default="1280x720")
    args = p.parse_args()
    rodar(args.modelo, args.camera, args.conf, args.resolucao)


if __name__ == "__main__":
    main()
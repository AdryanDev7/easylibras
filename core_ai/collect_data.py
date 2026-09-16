"""
Coleta de gestos pela webcam -> dataset/raw_landmarks.csv

Guardamos os landmarks CRUS (normalizados 0..1 pelo MediaPipe) + tamanho da
imagem + lado da mao. Toda a normalizacao acontece depois, no preprocess.
Assim voce pode mudar a normalizacao sem precisar recoletar nada.

Exemplos:
    python -m core_ai.collect_data --sessao marcos_dia1 --letras A,B,C
    python -m core_ai.collect_data --sessao arthur_noite --letras A --amostras 400

REGRA DE OURO: crie uma SESSAO nova a cada gravacao (pessoa/dia/luz/roupa
diferente). O treino separa treino e teste POR SESSAO -- e so assim a acuracia
que voce coloca no relatorio significa alguma coisa.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from .config import AMOSTRAS_PADRAO, RAW_CSV
from .hand_tracker import HandTracker, desenhar_mao

CABECALHO = (
    ["label", "sessao", "lado", "img_w", "img_h"]
    + [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
)


def _abrir_csv(caminho: Path):
    novo = not caminho.exists()
    arq = open(caminho, "a", newline="", encoding="utf-8")
    escritor = csv.writer(arq)
    if novo:
        escritor.writerow(CABECALHO)
    return arq, escritor


def _texto(frame, txt, y, cor=(255, 255, 255), escala=0.7):
    cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def coletar(letras: list[str], sessao: str, amostras: int, camera: int, saida: Path,
            resolucao: str = "1280x720"):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera {camera}.")
    larg, alt = (int(v) for v in resolucao.lower().split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, larg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alt)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (real_w, real_h) != (larg, alt):
        print(f"[info] A camera nao aceitou {larg}x{alt}; usando {real_w}x{real_h}.")

    arq, escritor = _abrir_csv(saida)
    tracker = HandTracker(num_maos=1, modo="video")
    t0 = time.time()

    try:
        for letra in letras:
            estado = "espera"   # espera -> contagem -> gravando
            coletadas = 0
            inicio_contagem = 0.0

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)  # espelho: mais natural para quem sinaliza
                h, w = frame.shape[:2]
                ts = int((time.time() - t0) * 1000)
                maos = tracker.processar(frame, ts)

                if maos:
                    desenhar_mao(frame, maos[0].pontos)

                if estado == "espera":
                    _texto(frame, f"Letra: {letra}   sessao: {sessao}", 32, (0, 255, 255), 0.9)
                    _texto(frame, "ESPACO = comecar   N = pular letra   ESC = sair", 64)
                    _texto(frame, "Varie angulo, distancia e altura da mao durante a gravacao", 92, (180, 180, 180), 0.55)
                elif estado == "contagem":
                    restante = 3 - int(time.time() - inicio_contagem)
                    if restante <= 0:
                        estado = "gravando"
                    else:
                        _texto(frame, f"Preparar... {restante}", 48, (0, 200, 255), 1.4)
                elif estado == "gravando":
                    if maos:
                        mao = maos[0]
                        pts = mao.pontos.copy()
                        pts[:, 0] /= w
                        pts[:, 1] /= h
                        pts[:, 2] /= w
                        escritor.writerow(
                            [letra, sessao, mao.lado, w, h]
                            + list(pts[:, 0]) + list(pts[:, 1]) + list(pts[:, 2])
                        )
                        coletadas += 1
                    _texto(frame, f"GRAVANDO {letra}: {coletadas}/{amostras}", 40, (0, 80, 255), 1.0)
                    barra = int((coletadas / amostras) * (w - 24))
                    cv2.rectangle(frame, (12, h - 34), (12 + barra, h - 14), (0, 80, 255), -1)
                    if not maos:
                        _texto(frame, "mao nao detectada", 72, (0, 0, 255))
                    if coletadas >= amostras:
                        arq.flush()
                        break

                cv2.imshow("EasyLibras - coleta", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == 27:
                    raise KeyboardInterrupt
                if tecla == ord(" ") and estado == "espera":
                    estado, inicio_contagem = "contagem", time.time()
                if tecla in (ord("n"), ord("N")) and estado == "espera":
                    break

            print(f"[ok] {letra}: {coletadas} amostras gravadas (sessao {sessao})")
    except KeyboardInterrupt:
        print("\nColeta interrompida. O que ja foi gravado esta salvo.")
    finally:
        arq.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Coleta de gestos de Libras")
    p.add_argument("--letras", required=True, help="Ex.: A,B,C")
    p.add_argument("--sessao", required=True, help="Identificador unico da gravacao, ex.: marcos_dia1")
    p.add_argument("--amostras", type=int, default=AMOSTRAS_PADRAO)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--saida", type=Path, default=RAW_CSV)
    p.add_argument("--resolucao", default="1280x720",
                   help="ex.: 640x480 (4:3 costuma dar um campo de visao mais largo)")
    args = p.parse_args()

    letras = [l.strip().upper() for l in args.letras.split(",") if l.strip()]
    coletar(letras, args.sessao, args.amostras, args.camera, args.saida, args.resolucao)


if __name__ == "__main__":
    main()
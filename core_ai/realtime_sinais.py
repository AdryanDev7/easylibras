"""
Reconhecimento AO VIVO de sinais dinamicos (palavras e sinais-nome).

    python -m core_ai.realtime_sinais
    python -m core_ai.realtime_sinais --manual     # voce marca inicio/fim com ESPACO
    python -m core_ai.realtime_sinais --debug      # mostra os pontos do rosto/corpo

Modo automatico (padrao): faca o sinal e abaixe as maos. Quando as maos voltam
para baixo da linha cinza, o trecho e cortado e classificado.

Teclas: ESC sai | C limpa o texto | B apaga a ultima palavra | ESPACO (modo manual)
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import joblib

from .config import SINAIS_MODEL_PATH
from .corpo_tracker import CorpoTracker, desenhar_quadro
from .features_sinais import (VERSAO_SINAIS, Referencial, referencial_padrao,
                              sequencia_para_matriz)
from .sinais import Segmentador


def carregar(caminho: Path):
    if not caminho.exists():
        raise SystemExit(f"Nao achei {caminho}. Rode: python -m core_ai.treinar_sinais")
    pacote = joblib.load(caminho)
    if pacote.get("versao_sinais") != VERSAO_SINAIS:
        raise SystemExit("Modelo de sinais de outra versao de features. Rode treinar_sinais de novo.")
    return pacote["classificador"]


def _texto(frame, txt, pos, cor=(255, 255, 255), escala=0.7, grossura=1):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grossura + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def rodar(caminho: Path, camera: int, resolucao: str, manual: bool, debug: bool):
    clf = carregar(caminho)
    print(f"Sinais conhecidos: {', '.join(clf.sinais)}")

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera {camera}.")
    larg, alt = (int(v) for v in resolucao.lower().split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, larg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alt)

    tracker = CorpoTracker()
    seg = Segmentador()
    # classificar leva ~0,1-0,5 s: roda em paralelo para o video nao travar
    executor = ThreadPoolExecutor(max_workers=1)
    pendente = None
    ultimo = None          # ultimo resultado, para mostrar na tela
    palavras: list[str] = []
    gravando_manual, buffer_manual = False, []
    ref = None
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)      # MESMO espelho da coleta
            h, w = frame.shape[:2]
            q = tracker.processar(frame, int((time.time() - t0) * 1000))
            if q.pose is not None:
                ref = Referencial(q.pose)
            ref_atual = ref or referencial_padrao(w, h)

            desenhar_quadro(frame, q, ref_atual.pontos_desenho() if debug and q.pose is not None else None)
            y_rep = int(ref_atual.origem[1] + seg.linha_repouso * ref_atual.escala)
            cv2.line(frame, (0, y_rep), (w, y_rep), (80, 80, 80), 1, cv2.LINE_AA)

            # --- corta o sinal ---
            trecho = None
            if manual:
                if gravando_manual:
                    buffer_manual.append(q)
            else:
                trecho = seg.atualizar(q, ref_atual)
            if trecho and pendente is None:
                pendente = executor.submit(lambda t=trecho: clf.prever(sequencia_para_matriz(t)))

            # --- resultado pronto? ---
            if pendente is not None and pendente.done():
                ultimo = pendente.result()
                pendente = None
                if ultimo["sinal"]:
                    palavras.append(ultimo["sinal"])
                print(f"-> {ultimo['sinal'] or '(desconhecido)'}  "
                      f"[mais parecido: {ultimo.get('mais_parecido')}, distancia "
                      f"{(ultimo.get('distancia') or 0):.2f} / limiar {ultimo.get('limiar', 0):.2f}]")

            # --- HUD ---
            cv2.rectangle(frame, (0, 0), (w, 96), (20, 20, 20), -1)
            gravando = gravando_manual if manual else seg.gravando
            if gravando:
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            if pendente is not None:
                _texto(frame, "reconhecendo...", (20, 60), (0, 200, 255), 1.0, 2)
            elif ultimo:
                if ultimo["sinal"]:
                    _texto(frame, ultimo["sinal"], (20, 62), (0, 220, 120), 1.4, 3)
                    _texto(frame, f"confianca {ultimo['confianca'] * 100:.0f}%", (20, 88), (170, 170, 170), 0.5)
                else:
                    _texto(frame, f"? (parece {ultimo.get('mais_parecido')})", (20, 62),
                           (60, 160, 255), 1.0, 2)
            else:
                dica = "ESPACO grava/para" if manual else "faca o sinal e abaixe as maos"
                _texto(frame, dica, (20, 60), (150, 150, 150), 0.8)
            if q.pose is None:
                _texto(frame, "corpo nao detectado: rosto e ombros precisam aparecer",
                       (20, h - 100), (0, 0, 255), 0.6)

            cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
            _texto(frame, (" ".join(palavras))[-50:] or "_", (20, h - 32), (255, 255, 255), 1.0, 2)
            _texto(frame, "ESC sair | C limpar | B apagar", (w - 300, h - 8), (140, 140, 140), 0.5)

            cv2.imshow("EasyLibras - sinais", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            if tecla in (ord("c"), ord("C")):
                palavras, ultimo = [], None
            if tecla in (ord("b"), ord("B")) and palavras:
                palavras.pop()
            if tecla == ord(" ") and manual:
                if gravando_manual and pendente is None:
                    t = buffer_manual
                    pendente = executor.submit(lambda t=t: clf.prever(sequencia_para_matriz(t)))
                gravando_manual, buffer_manual = not gravando_manual, []
    finally:
        executor.shutdown(wait=False)
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTexto final: {' '.join(palavras)!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", type=Path, default=SINAIS_MODEL_PATH)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--resolucao", default="1280x720")
    p.add_argument("--manual", action="store_true", help="ESPACO marca inicio e fim do sinal")
    p.add_argument("--debug", action="store_true", help="mostra olho, bochecha, peito... na tela")
    args = p.parse_args()
    rodar(args.modelo, args.camera, args.resolucao, args.manual, args.debug)


if __name__ == "__main__":
    main()

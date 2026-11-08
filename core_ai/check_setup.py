"""
Verificacao do ambiente. RODE ISSO PRIMEIRO.

    python -m core_ai.check_setup

Confere versao do Python, bibliotecas, baixa o modelo do MediaPipe e abre a
camera mostrando o esqueleto da mao. Se este script funcionar, o resto funciona.
"""
from __future__ import annotations

import sys
import time


def checar_versao():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("  [ERRO] Use Python 3.10 ou superior.")
        return False
    if v >= (3, 13):
        print("  [ERRO] O mediapipe ainda nao publica wheel para Python 3.13+.")
        print("         Crie o venv com python3.12 (ou 3.11 / 3.10).")
        return False
    print("  ok")
    return True


def checar_libs():
    ok = True
    for nome in ["cv2", "mediapipe", "numpy", "pandas", "sklearn", "joblib"]:
        try:
            mod = __import__(nome)
            print(f"{nome}: {getattr(mod, '__version__', '?')}  ok")
        except Exception as exc:
            print(f"{nome}: FALHOU -> {exc}")
            ok = False

    try:
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            print("[info] Sua versao do mediapipe ainda tem o modulo antigo `solutions`. "
                  "Este projeto usa a API nova (tasks) de proposito -- funciona igual.")
        else:
            print("[info] `mp.solutions` nao existe nesta versao (esperado). "
                  "Codigo antigo com `mp.solutions.hands` vai quebrar; o nosso nao usa isso.")
    except Exception:
        pass
    return ok


def checar_camera(indice: int = 0):
    import cv2
    from .hand_tracker import HandTracker, desenhar_mao

    print("\nBaixando/carregando modelo do MediaPipe...")
    tracker = HandTracker(num_maos=2, modo="video")

    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        print(f"[ERRO] Camera {indice} nao abriu. Tente --camera 1, ou feche outros "
              "programas que estejam usando a webcam.")
        return False

    print("Camera aberta. Mostre a mao. ESC para sair.\n")
    t0 = time.time()
    frames, com_mao = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            maos = tracker.processar(frame, int((time.time() - t0) * 1000))
            frames += 1
            if maos:
                com_mao += 1
                desenhar_mao(frame, maos[0].pontos)
                txt = f"mao detectada ({maos[0].lado}, {maos[0].score:.2f})"
                cor = (0, 220, 120)
            else:
                txt, cor = "nenhuma mao", (60, 60, 255)
            cv2.putText(frame, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2, cv2.LINE_AA)
            cv2.putText(frame, "ESC para sair", (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("EasyLibras - teste de setup", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Frames: {frames} | com mao detectada: {com_mao}")
    if frames and com_mao / frames < 0.2:
        print("[aviso] Pouca deteccao. Melhore a iluminacao e afaste a mao do fundo.")
    return True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--sem-camera", action="store_true")
    args = p.parse_args()

    print("=== EasyLibras: verificacao de ambiente ===\n")
    if not checar_versao():
        sys.exit(1)
    print()
    if not checar_libs():
        print("\nInstale o que faltou: pip install -r requirements.txt")
        sys.exit(1)
    if not args.sem_camera:
        checar_camera(args.camera)
    print("\nTudo certo. Proximo passo: python -m core_ai.collect_data --letras A,B,C --sessao sua_sessao")


if __name__ == "__main__":
    main()

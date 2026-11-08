"""
Gravacao de sinais DINAMICOS (palavras e sinais-nome) -> dataset/sinais/<SINAL>/*.npz

    python -m core_ai.coletar_sinais --sinais OCULOS,SAUDADE,NOME_ADRYAN --sessao adryan_1
    python -m core_ai.coletar_sinais --sinais _NADA --sessao adryan_1 --repeticoes 10

Cada gravacao e UM sinal completo. Fluxo na tela:
    ESPACO  comeca (contagem de 2 s) / termina a gravacao antes da hora
    D       descarta a ultima gravacao
    N       pula para o proximo sinal
    ESC     sai (o que ja foi gravado fica salvo)

A gravacao tambem termina sozinha quando voce abaixa as maos (posicao de
repouso) -- do mesmo jeito que o reconhecimento ao vivo vai cortar os sinais.

DICAS QUE FAZEM DIFERENCA:
  - comece e termine com as maos ABAIXADAS, fora da frente do peito
  - enquadre do quadril para cima, com o rosto inteiro na tela
  - 5 repeticoes por sinal por pessoa ja funciona; varie um pouco a velocidade
  - grave tambem um sinal "_NADA": cocar o rosto, ajeitar o cabelo, ajeitar o
    oculos... Sao gestos que o sistema deve IGNORAR (nomes com "_" no inicio)
  - para o relatorio, grave cada sinal com 2+ pessoas (sessoes diferentes)
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .config import SINAIS_DIR, SINAIS_MAX_SEGUNDOS
from .corpo_tracker import CorpoTracker, desenhar_quadro, quadros_para_bruto
from .features_sinais import Referencial, referencial_padrao
from .sinais import Segmentador


def _texto(frame, txt, y, cor=(255, 255, 255), escala=0.7, x=12):
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def nome_valido(sinal: str) -> str:
    s = sinal.strip().upper().replace(" ", "_")
    s = s.replace("Ç", "C").replace("Ã", "A").replace("Á", "A").replace("É", "E") \
         .replace("Ê", "E").replace("Í", "I").replace("Ó", "O").replace("Ô", "O").replace("Ú", "U")
    if not re.fullmatch(r"_?[A-Z0-9_]+", s):
        raise SystemExit(f"Nome de sinal invalido: {sinal!r} (use letras, numeros e _)")
    return s


def contar(sinal: str) -> int:
    pasta = SINAIS_DIR / sinal
    return len(list(pasta.glob("*.npz"))) if pasta.exists() else 0


def salvar(quadros, sinal: str, sessao: str) -> Path:
    pasta = SINAIS_DIR / sinal
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{sessao}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.npz"
    np.savez_compressed(destino, sinal=sinal, sessao=sessao, **quadros_para_bruto(quadros))
    return destino


def coletar(sinais: list[str], sessao: str, repeticoes: int, camera: int, resolucao: str):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera {camera}.")
    larg, alt = (int(v) for v in resolucao.lower().split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, larg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alt)

    print("Carregando modelos do MediaPipe (maos + pose)...")
    tracker = CorpoTracker()
    t0 = time.time()
    ultimo_salvo: Path | None = None
    ref = None

    try:
        for sinal in sinais:
            feitas = 0
            while feitas < repeticoes:
                estado, inicio = "espera", 0.0
                gravados = []
                seg = Segmentador()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)   # MESMO espelho da coleta do alfabeto
                    h, w = frame.shape[:2]
                    q = tracker.processar(frame, int((time.time() - t0) * 1000))
                    if q.pose is not None:
                        ref = Referencial(q.pose)
                    ref_atual = ref or referencial_padrao(w, h)
                    desenhar_quadro(frame, q, ref_atual.pontos_desenho() if q.pose is not None else None)

                    # linha de repouso: abaixo dela, as maos "nao estao sinalizando"
                    y_rep = int(ref_atual.origem[1] + seg.linha_repouso * ref_atual.escala)
                    cv2.line(frame, (0, y_rep), (w, y_rep), (80, 80, 80), 1, cv2.LINE_AA)

                    total = contar(sinal)
                    _texto(frame, f"Sinal: {sinal}   ({feitas}/{repeticoes} nesta sessao, {total} no total)",
                           32, (0, 255, 255), 0.8)
                    if q.pose is None:
                        _texto(frame, "corpo nao detectado: afaste-se, rosto e ombros na tela",
                               h - 20, (0, 0, 255), 0.6)

                    if estado == "espera":
                        _texto(frame, "ESPACO gravar | D descartar ultima | N proximo sinal | ESC sair", 62)
                        _texto(frame, "Comece e termine com as maos abaixo da linha cinza", 88,
                               (180, 180, 180), 0.55)
                    elif estado == "contagem":
                        falta = 2 - int(time.time() - inicio)
                        if falta <= 0:
                            estado, inicio = "gravando", time.time()
                        else:
                            _texto(frame, f"Preparar... {falta}", 70, (0, 200, 255), 1.4)
                    elif estado == "gravando":
                        gravados.append(q)
                        terminou = seg.atualizar(q, ref_atual) is not None
                        dur = time.time() - inicio
                        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
                        _texto(frame, f"GRAVANDO {sinal}  {dur:.1f}s  (ESPACO termina)", 70,
                               (0, 80, 255), 0.9)
                        if terminou or dur > SINAIS_MAX_SEGUNDOS:
                            break

                    cv2.imshow("EasyLibras - coleta de sinais", frame)
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla == 27:
                        raise KeyboardInterrupt
                    if tecla == ord(" "):
                        if estado == "espera":
                            estado, inicio = "contagem", time.time()
                        elif estado == "gravando":
                            break
                    if tecla in (ord("d"), ord("D")) and estado == "espera" and ultimo_salvo:
                        ultimo_salvo.unlink(missing_ok=True)
                        print(f"[descartado] {ultimo_salvo.name}")
                        ultimo_salvo = None
                        feitas = max(0, feitas - 1)
                    if tecla in (ord("n"), ord("N")) and estado == "espera":
                        feitas = repeticoes
                        break

                com_mao = sum(1 for g in gravados if g.maos)
                if gravados and com_mao >= 5:
                    ultimo_salvo = salvar(gravados, sinal, sessao)
                    feitas += 1
                    print(f"[ok] {sinal} #{feitas}: {len(gravados)} frames -> {ultimo_salvo.name}")
                elif gravados:
                    print(f"[ignorado] {sinal}: so {com_mao} frames com mao. Repita.")
    except KeyboardInterrupt:
        print("\nColeta interrompida. O que ja foi gravado esta salvo.")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Coleta de sinais dinamicos de Libras")
    p.add_argument("--sinais", required=True, help="Ex.: OCULOS,SAUDADE,NOME_ADRYAN")
    p.add_argument("--sessao", required=True, help="pessoa + dia, ex.: adryan_1")
    p.add_argument("--repeticoes", type=int, default=5)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--resolucao", default="1280x720")
    args = p.parse_args()
    sinais = [nome_valido(s) for s in args.sinais.split(",") if s.strip()]
    coletar(sinais, args.sessao, args.repeticoes, args.camera, args.resolucao)


if __name__ == "__main__":
    main()

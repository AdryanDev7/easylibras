"""
Converte o data.pickle antigo para dataset/raw_landmarks.csv.

    python -m tools.importar_pickle --arquivo data.pickle --sessao legacy_adryan

O formato antigo (do tutorial em que o projeto foi baseado) guarda por amostra
42 numeros: x0,y0,x1,y1,...,x20,y20 -- ou seja, SO x e y, sem z, sem saber qual
mao era e sem resolucao da imagem. Cada ponto ja vem transladado pelo canto da
caixa da mao.

O que a conversao consegue recuperar e o que nao consegue:

  ok    a geometria da mao (a translacao antiga e desfeita pela normalizacao nova)
  ok    os rotulos das letras
  falta z  -> preenchido com 0. Dados novos terao z real.
  falta lado -> assumido pelo parametro --lado
  falta resolucao -> assumida pelo parametro --resolucao

Por isso: use para validar o pipeline hoje, sem camera. Para o modelo final,
recolete com core_ai.collect_data.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from core_ai.collect_data import CABECALHO
from core_ai.config import RAW_CSV


def converter(arquivo: Path, sessao: str, lado: str, largura: int, altura: int,
              saida: Path) -> int:
    with open(arquivo, "rb") as f:
        dados = pickle.load(f)

    if not isinstance(dados, dict) or "data" not in dados or "labels" not in dados:
        raise SystemExit(
            f"Formato inesperado. Esperava um dict com 'data' e 'labels', "
            f"achei: {type(dados)}"
        )

    X = np.asarray(dados["data"], dtype=np.float32)
    y = [str(v) for v in dados["labels"]]

    if X.ndim != 2 or X.shape[1] != 42:
        raise SystemExit(
            f"Esperava 42 colunas (21 pontos x, y), achei {X.shape}. "
            "Me manda o arquivo que eu ajusto o conversor."
        )

    pts = X.reshape(len(X), 21, 2)  # intercalado x0,y0,x1,y1,...

    novo = not saida.exists()
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(CABECALHO)
        for i, rotulo in enumerate(y):
            xs = pts[i, :, 0]
            ys = pts[i, :, 1]
            zs = np.zeros(21, dtype=np.float32)   # o formato antigo nao tem z
            w.writerow([rotulo, sessao, lado, largura, altura]
                       + list(xs) + list(ys) + list(zs))

    from collections import Counter
    print(f"Convertidas {len(y)} amostras -> {saida}")
    for letra, n in sorted(Counter(y).items()):
        print(f"  {letra}: {n}")
    print(f"\nSessao: {sessao} | lado assumido: {lado} | resolucao assumida: {largura}x{altura}")
    return len(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arquivo", type=Path, required=True)
    p.add_argument("--sessao", default="legacy", help="nome da sessao antiga")
    p.add_argument("--lado", default="Right", choices=["Right", "Left"],
                   help="qual mao foi usada na gravacao antiga")
    p.add_argument("--resolucao", default="640x480",
                   help="resolucao da camera na gravacao antiga, ex.: 640x480")
    p.add_argument("--saida", type=Path, default=RAW_CSV)
    args = p.parse_args()

    largura, altura = (int(v) for v in args.resolucao.lower().split("x"))
    converter(args.arquivo, args.sessao, args.lado, largura, altura, args.saida)


if __name__ == "__main__":
    main()

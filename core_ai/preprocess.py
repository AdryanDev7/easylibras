"""
dataset/raw_landmarks.csv (+ dataset/externos/*.csv) -> dataset/processed.npz

Aplica a normalizacao geometrica e guarda tambem o grupo (sessao) de cada
amostra, que o treino usa para fazer uma divisao honesta entre treino e teste.

    python -m core_ai.preprocess                 # seus dados + externos
    python -m core_ai.preprocess --sem-externos  # so o que voces gravaram
    python -m core_ai.preprocess --entrada a.csv b.csv

Os datasets externos (ex.: Kaggle) entram por tools/importar_dataset.py, que
grava em dataset/externos/. Apagar o arquivo de la tira ele do treino.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_DIR, PROCESSED_NPZ, RAW_CSV
from .features import VERSAO_FEATURES, normalizar_pontos


DIR_EXTERNOS = DATASET_DIR / "externos"


def arquivos_padrao(com_externos: bool = True) -> list[Path]:
    arquivos = [RAW_CSV] if RAW_CSV.exists() else []
    if com_externos and DIR_EXTERNOS.exists():
        arquivos += sorted(DIR_EXTERNOS.glob("*.csv"))
    return arquivos


def carregar(caminhos: list[Path] | Path) -> pd.DataFrame:
    if isinstance(caminhos, Path):
        caminhos = [caminhos]
    if not caminhos:
        raise SystemExit(
            "Nenhum dado encontrado. Grave com `python -m core_ai.collect_data` "
            "ou importe um dataset com `python -m tools.importar_dataset`."
        )
    partes = []
    for c in caminhos:
        if not c.exists():
            raise SystemExit(f"Nao achei {c}.")
        df = pd.read_csv(c)
        print(f"  {c.name}: {len(df)} amostras")
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    if df.empty:
        raise SystemExit("CSV vazio.")
    return df


def processar(df: pd.DataFrame):
    cols_x = [f"x{i}" for i in range(21)]
    cols_y = [f"y{i}" for i in range(21)]
    cols_z = [f"z{i}" for i in range(21)]

    X = np.stack(
        [df[cols_x].to_numpy(np.float32),
         df[cols_y].to_numpy(np.float32),
         df[cols_z].to_numpy(np.float32)],
        axis=-1,
    )  # (N, 21, 3) ainda em 0..1

    largura = df["img_w"].to_numpy(np.float32)[:, None]
    altura = df["img_h"].to_numpy(np.float32)[:, None]

    # volta para escala de pixels -- sem isso a proporcao da imagem (16:9)
    # deforma a mao e o modelo aprende a distorcao junto com o gesto
    X[:, :, 0] *= largura
    X[:, :, 1] *= altura
    X[:, :, 2] *= largura

    lados = df["lado"].astype(str).to_numpy()
    X_norm = np.stack([normalizar_pontos(X[i], lados[i]) for i in range(len(X))])

    y = df["label"].astype(str).to_numpy()
    grupos = df["sessao"].astype(str).to_numpy()
    return X_norm, y, grupos


def relatorio(y, grupos):
    print("\n=== Dataset ===")
    print(f"Amostras: {len(y)} | Classes: {len(set(y))} | Sessoes: {len(set(grupos))}")
    contagem = Counter(y)
    for letra, n in sorted(contagem.items()):
        sessoes = len(set(grupos[y == letra]))
        alerta = "  <-- poucas sessoes" if sessoes < 2 else ""
        print(f"  {letra}: {n:5d} amostras em {sessoes} sessao(oes){alerta}")

    por_sessao = Counter(grupos)
    if len(por_sessao) <= 20:
        print("\nPor sessao:")
        for sessao, n in sorted(por_sessao.items()):
            print(f"  {sessao}: {n}")

    menor, maior = min(contagem.values()), max(contagem.values())
    if maior > 2 * menor:
        print(f"\n[aviso] Classes desbalanceadas ({menor} vs {maior}). "
              "Colete mais da letra com menos amostras.")
    if len(set(grupos)) < 2:
        print("\n[AVISO IMPORTANTE] So existe 1 sessao. A acuracia do treino vai sair "
              "inflada (frames vizinhos sao quase identicos). Grave pelo menos 3 sessoes, "
              "de preferencia com pessoas/luzes diferentes.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", type=Path, nargs="+", default=None,
                   help="CSVs de entrada (padrao: raw_landmarks.csv + dataset/externos/*.csv)")
    p.add_argument("--sem-externos", action="store_true",
                   help="ignora dataset/externos/ (usa so o que voces gravaram)")
    p.add_argument("--saida", type=Path, default=PROCESSED_NPZ)
    args = p.parse_args()

    entradas = args.entrada or arquivos_padrao(not args.sem_externos)
    print("Lendo:")
    df = carregar(entradas)
    X_norm, y, grupos = processar(df)
    relatorio(y, grupos)

    np.savez_compressed(args.saida, X=X_norm, y=y, grupos=grupos, versao=VERSAO_FEATURES)
    print(f"\nSalvo em {args.saida}  (X={X_norm.shape})")


if __name__ == "__main__":
    main()

"""
Treino do classificador de letras.

    python -m core_ai.train

Metodologia (isso vale nota no trabalho academico):
- a divisao treino/teste e POR SESSAO, nunca por frame. Frames vizinhos de uma
  mesma gravacao sao quase identicos; dividir aleatoriamente vaza o teste para
  dentro do treino e produz 99% de acuracia falsa.
- o data augmentation e aplicado SOMENTE no treino.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from .config import CLASSIFIER_PATH, PROCESSED_NPZ
from .features import VERSAO_FEATURES, _features_de_normalizados, aumentar_lote


def dividir(X, y, grupos, semente=42):
    """Separa por sessao quando da; senao avisa e divide por frame."""
    n_sessoes = len(set(grupos))
    if n_sessoes >= 2:
        divisor = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semente)
        tr, te = next(divisor.split(X, y, groups=grupos))
        print(f"Split POR SESSAO -> treino: {sorted(set(grupos[tr]))} | teste: {sorted(set(grupos[te]))}")
        return tr, te

    print("[AVISO] Apenas 1 sessao no dataset. Dividindo por frame: as metricas "
          "abaixo sao OTIMISTAS e nao representam desempenho real.")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=semente, stratify=y)
    return tr, te


def treinar(X, y, grupos, copias_aug=3, arvores=400, semente=42):
    idx_tr, idx_te = dividir(X, y, grupos, semente)

    X_tr, y_tr = aumentar_lote(X[idx_tr], y[idx_tr], copias=copias_aug, seed=semente)
    X_te = np.array([_features_de_normalizados(p) for p in X[idx_te]], dtype=np.float32)
    y_te = y[idx_te]

    print(f"\nTreino: {len(y_tr)} amostras (com augmentation) | Teste: {len(y_te)}")

    modelo = RandomForestClassifier(
        n_estimators=arvores,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=semente,
    )
    modelo.fit(X_tr, y_tr)

    pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"\n=== Acuracia no teste: {acc:.3f} ===\n")
    print(classification_report(y_te, pred, zero_division=0))

    rotulos = sorted({str(v) for v in y})
    mc = confusion_matrix(y_te, pred, labels=rotulos)
    confusoes = []
    for i, real in enumerate(rotulos):
        for j, previsto in enumerate(rotulos):
            if i != j and mc[i, j] > 0:
                confusoes.append((mc[i, j], real, previsto))
    if confusoes:
        print("Maiores confusoes (real -> previsto):")
        for n, real, previsto in sorted(confusoes, reverse=True)[:8]:
            print(f"  {real} -> {previsto}: {n}")
        print("  (letras que se confundem muito pedem mais dados OU sao dinamicas)")

    return modelo, acc, rotulos, mc


def validacao_cruzada(X, y, grupos, arvores=300, semente=42):
    """GroupKFold: numero mais confiavel para colocar no relatorio."""
    n_sessoes = len(set(grupos))
    if n_sessoes < 3:
        print("\n[info] Validacao cruzada por sessao exige >= 3 sessoes. Pulando.")
        return None
    k = min(5, n_sessoes)
    gkf = GroupKFold(n_splits=k)
    scores = []
    for tr, te in gkf.split(X, y, groups=grupos):
        X_tr, y_tr = aumentar_lote(X[tr], y[tr], copias=2, seed=semente)
        X_te = np.array([_features_de_normalizados(p) for p in X[te]], dtype=np.float32)
        m = RandomForestClassifier(n_estimators=arvores, class_weight="balanced",
                                   n_jobs=-1, random_state=semente).fit(X_tr, y_tr)
        scores.append(accuracy_score(y[te], m.predict(X_te)))
    scores = np.array(scores)
    print(f"\n=== GroupKFold ({k} folds): {scores.mean():.3f} +/- {scores.std():.3f} ===")
    print("    (use ESTE numero no relatorio)")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", type=Path, default=PROCESSED_NPZ)
    p.add_argument("--saida", type=Path, default=CLASSIFIER_PATH)
    p.add_argument("--arvores", type=int, default=400)
    p.add_argument("--aug", type=int, default=3, help="copias sinteticas por amostra")
    p.add_argument("--cv", action="store_true", help="roda validacao cruzada por sessao")
    args = p.parse_args()

    if not args.entrada.exists():
        raise SystemExit(f"Nao achei {args.entrada}. Rode: python -m core_ai.preprocess")

    dados = np.load(args.entrada, allow_pickle=True)
    X, y, grupos = dados["X"], dados["y"].astype(str), dados["grupos"].astype(str)

    modelo, acc, rotulos, mc = treinar(X, y, grupos, args.aug, args.arvores)
    scores_cv = validacao_cruzada(X, y, grupos, args.arvores) if args.cv else None

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "rotulos": rotulos,
            "versao_features": VERSAO_FEATURES,
            "acuracia_teste": float(acc),
            "acuracia_cv": None if scores_cv is None else float(scores_cv.mean()),
            "n_amostras": int(len(y)),
            "sessoes": sorted({str(g) for g in grupos}),
        },
        args.saida,
    )
    print(f"\nModelo salvo em {args.saida}")

    relatorio = args.saida.with_suffix(".metrics.json")
    relatorio.write_text(json.dumps({
        "acuracia_teste": float(acc),
        "acuracia_cv": None if scores_cv is None else float(scores_cv.mean()),
        "rotulos": rotulos,
        "matriz_confusao": mc.tolist(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metricas salvas em {relatorio}")


if __name__ == "__main__":
    main()

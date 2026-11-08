"""
Treino do classificador de letras.

    python -m core_ai.train

Metodologia (isso vale nota no trabalho academico):
- a divisao treino/teste e POR SESSAO, nunca por frame. Frames vizinhos de uma
  mesma gravacao sao quase identicos; dividir aleatoriamente vaza o teste para
  dentro do treino e produz 99% de acuracia falsa.
- o data augmentation e aplicado SOMENTE no treino.

Com dataset externo (Kaggle), a pergunta que importa e: "um modelo treinado
com os dados dos outros reconhece a NOSSA mao, na NOSSA camera?". Para medir
isso, mantenha o externo sempre no treino e teste so nas sessoes de voces:

    python -m core_ai.train --cv --sempre-treino "kaggle_*"

Ou escolha explicitamente quem fica no teste:

    python -m core_ai.train --teste-sessoes adryan_1,enzo_1
"""
from __future__ import annotations

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from .config import CLASSIFIER_PATH, PROCESSED_NPZ
from .features import VERSAO_FEATURES, _features_de_normalizados, aumentar_lote


def _nomes(arr) -> list[str]:
    return sorted({str(v) for v in arr})


def _casa(grupos: np.ndarray, padroes: list[str] | None) -> np.ndarray:
    """Mascara booleana: quais amostras pertencem a sessoes que casam com os padroes."""
    if not padroes:
        return np.zeros(len(grupos), dtype=bool)
    unicos = {g: any(fnmatch(g, pd) for pd in padroes) for g in set(grupos)}
    return np.array([unicos[g] for g in grupos], dtype=bool)


def dividir(X, y, grupos, semente=42, teste=None, sempre_treino=None):
    """Separa por sessao quando da; senao avisa e divide por frame.

    teste:          padroes (glob) das sessoes que vao para o teste.
    sempre_treino:  padroes das sessoes que NUNCA vao para o teste (ex.: "kaggle_*").
    """
    todos = np.arange(len(y))

    if teste:
        m = _casa(grupos, teste)
        if not m.any():
            raise SystemExit(f"Nenhuma sessao casa com --teste-sessoes {teste}. "
                             f"Sessoes existentes: {_nomes(grupos)}")
        if m.all():
            raise SystemExit("--teste-sessoes pegou TODAS as sessoes; nao sobra nada para treino.")
        tr, te = todos[~m], todos[m]
        print(f"Split MANUAL -> teste: {_nomes(grupos[te])}")
        return tr, te

    fixo = _casa(grupos, sempre_treino)
    candidatos = todos[~fixo]
    sessoes_cand = _nomes(grupos[candidatos])

    if sempre_treino and not fixo.any():
        print(f"[aviso] Nenhuma sessao casa com --sempre-treino {sempre_treino}.")

    if fixo.any() and len(sessoes_cand) == 1:
        # caso classico: treina no externo, testa na unica sessao de voces
        print(f"Split EXTERNO -> treino: {_nomes(grupos[fixo])} | teste: {sessoes_cand}")
        return todos[fixo], candidatos

    if len(sessoes_cand) >= 2:
        divisor = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semente)
        tr_c, te_c = next(divisor.split(candidatos, y[candidatos], groups=grupos[candidatos]))
        tr = np.concatenate([todos[fixo], candidatos[tr_c]])
        te = candidatos[te_c]
        print(f"Split POR SESSAO -> treino: {_nomes(grupos[tr])} | teste: {_nomes(grupos[te])}")
        return tr, te

    print("[AVISO] Apenas 1 sessao disponivel para teste. Dividindo por frame: as metricas "
          "abaixo sao OTIMISTAS e nao representam desempenho real.")
    base = candidatos if len(candidatos) else todos
    _, contagem = np.unique(y[base], return_counts=True)
    estratificar = y[base] if contagem.min() >= 2 else None
    tr_b, te = train_test_split(base, test_size=0.25, random_state=semente, stratify=estratificar)
    tr = np.concatenate([todos[fixo], tr_b]) if len(candidatos) else tr_b
    return tr, te


def treinar(X, y, grupos, copias_aug=3, arvores=400, semente=42, teste=None, sempre_treino=None):
    idx_tr, idx_te = dividir(X, y, grupos, semente, teste, sempre_treino)

    sem_treino = sorted(set(y[idx_te]) - set(y[idx_tr]))
    if sem_treino:
        print(f"[aviso] Letras no teste que NAO aparecem no treino: {sem_treino} "
              "(vao sair sempre erradas)")

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
    acc_por_sessao = {}
    for g in _nomes(grupos[idx_te]):
        m = grupos[idx_te] == g
        acc_por_sessao[str(g)] = float(accuracy_score(y_te[m], pred[m]))
    if len(acc_por_sessao) > 1:
        print("Acuracia por sessao de teste:")
        for g, a in acc_por_sessao.items():
            print(f"  {g}: {a:.3f}")
        print()

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

    return modelo, acc, rotulos, mc, acc_por_sessao


def validacao_cruzada(X, y, grupos, arvores=300, semente=42, sempre_treino=None):
    """GroupKFold: numero mais confiavel para colocar no relatorio.

    Sessoes em `sempre_treino` entram no treino de TODOS os folds e nunca sao
    testadas -- assim o numero mede o desempenho nas sessoes de voces.
    """
    fixo = _casa(grupos, sempre_treino)
    todos = np.arange(len(y))
    candidatos = todos[~fixo]
    n_sessoes = len(set(grupos[candidatos]))
    minimo = 2 if fixo.any() else 3
    if n_sessoes < minimo:
        print(f"\n[info] Validacao cruzada por sessao exige >= {minimo} sessoes testaveis "
              f"(tem {n_sessoes}). Pulando.")
        return None
    k = min(5, n_sessoes)
    gkf = GroupKFold(n_splits=k)
    scores = []
    for tr_c, te_c in gkf.split(candidatos, y[candidatos], groups=grupos[candidatos]):
        tr = np.concatenate([todos[fixo], candidatos[tr_c]])
        te = candidatos[te_c]
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
    p.add_argument("--so-avaliar", action="store_true",
                   help="salva o modelo do split de treino em vez de retreinar com tudo")
    p.add_argument("--teste-sessoes", default=None,
                   help="sessoes (separadas por virgula, aceita *) que vao para o teste")
    p.add_argument("--sempre-treino", default=None,
                   help='sessoes que nunca vao para o teste, ex.: "kaggle_*"')
    args = p.parse_args()

    def _lista(txt):
        return [t.strip() for t in txt.split(",") if t.strip()] if txt else None
    teste, sempre_treino = _lista(args.teste_sessoes), _lista(args.sempre_treino)

    if not args.entrada.exists():
        raise SystemExit(f"Nao achei {args.entrada}. Rode: python -m core_ai.preprocess")

    dados = np.load(args.entrada, allow_pickle=True)
    X, y, grupos = dados["X"], dados["y"].astype(str), dados["grupos"].astype(str)

    modelo, acc, rotulos, mc, acc_sessao = treinar(X, y, grupos, args.aug, args.arvores,
                                                   teste=teste, sempre_treino=sempre_treino)
    scores_cv = (validacao_cruzada(X, y, grupos, args.arvores, sempre_treino=sempre_treino)
                 if args.cv else None)

    # O modelo SALVO e treinado com tudo (treino + teste): o split acima serve
    # so para medir. Para a apresentacao, quanto mais dado, melhor.
    if not args.so_avaliar:
        print("\nRetreinando o modelo final com TODAS as amostras...")
        X_all, y_all = aumentar_lote(X, y, copias=args.aug, seed=42)
        modelo = RandomForestClassifier(n_estimators=args.arvores, class_weight="balanced",
                                        n_jobs=-1, random_state=42).fit(X_all, y_all)

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
            "acuracia_por_sessao_teste": acc_sessao,
            "treinado_com_tudo": not args.so_avaliar,
        },
        args.saida,
    )
    print(f"\nModelo salvo em {args.saida}")

    relatorio = args.saida.with_suffix(".metrics.json")
    relatorio.write_text(json.dumps({
        "acuracia_teste": float(acc),
        "acuracia_cv": None if scores_cv is None else float(scores_cv.mean()),
        "acuracia_por_sessao_teste": acc_sessao,
        "rotulos": rotulos,
        "matriz_confusao": mc.tolist(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metricas salvas em {relatorio}")


if __name__ == "__main__":
    main()

"""
Treina (e AVALIA) o reconhecedor de sinais dinamicos.

    python -m core_ai.treinar_sinais
    python -m core_ai.treinar_sinais --sem-espelho     # sem a copia "canhota"

Le dataset/sinais/<SINAL>/*.npz e salva models/sinais_knn.joblib.

Avaliacao: deixa UMA SESSAO (pessoa/dia) de fora por vez, treina com o resto e
testa nela -- igual ao GroupKFold do alfabeto. O numero que sai e "quao bem o
sistema reconhece alguem que ele nunca viu", que e o que importa.
Com uma sessao so, cai para "deixa uma gravacao de fora" (otimista; avisa).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from .config import SINAIS_DIR, SINAIS_MODEL_PATH
from .corpo_tracker import bruto_para_quadros
from .features_sinais import VERSAO_SINAIS, espelhar_quadros, sequencia_para_matriz
from .sinais import ClassificadorSinais

SUFIXO_ESPELHO = "~espelho"


def carregar_gravacoes(pasta: Path = SINAIS_DIR):
    """-> lista de (sinal, sessao, arquivo, quadros)."""
    if not pasta.exists():
        raise SystemExit(f"Nao achei {pasta}. Grave com: python -m core_ai.coletar_sinais")
    saida = []
    for arq in sorted(pasta.glob("*/*.npz")):
        with np.load(arq, allow_pickle=False) as d:
            bruto = {k: d[k] for k in d.files}
        saida.append((str(bruto["sinal"]), str(bruto["sessao"]), arq.name, bruto_para_quadros(bruto)))
    if not saida:
        raise SystemExit(f"Nenhuma gravacao em {pasta}.")
    return saida


def montar(gravacoes, espelho: bool):
    """-> matrizes, sinais, sessoes, origens (com copias espelhadas se pedido)."""
    M, S, G, O = [], [], [], []
    for sinal, sessao, arq, quadros in gravacoes:
        m = sequencia_para_matriz(quadros)
        if len(m) < 3:
            print(f"  [pulado] {arq}: quase sem mao detectada")
            continue
        M.append(m); S.append(sinal); G.append(sessao); O.append(arq)
        if espelho:
            M.append(sequencia_para_matriz(espelhar_quadros(quadros)))
            S.append(sinal); G.append(sessao + SUFIXO_ESPELHO); O.append(arq + " (espelhado)")
    return M, S, G, O


def relatorio(sinais, sessoes):
    print("\n=== Gravacoes ===")
    por = defaultdict(set)
    for s, g in zip(sinais, sessoes):
        por[s].add(g)
    cont = Counter(sinais)
    for s in sorted(cont):
        alerta = "  <-- so 1 sessao" if len(por[s]) < 2 else ""
        alerta = "  <-- poucas gravacoes (grave 3+)" if cont[s] < 3 else alerta
        print(f"  {s:20s} {cont[s]:3d} gravacoes em {len(por[s])} sessao(oes){alerta}")


def avaliar(M, S, G, O, k: int, fator: float):
    base = [g.removesuffix(SUFIXO_ESPELHO) for g in G]
    reais = [i for i, g in enumerate(G) if not g.endswith(SUFIXO_ESPELHO)]
    sessoes = sorted({base[i] for i in reais})

    resultados = []   # (verdadeiro, previsto, conhecido, sessao)
    if len(sessoes) >= 2:
        print(f"\nAvaliacao: deixa 1 sessao de fora ({len(sessoes)} rodadas)")
        for teste in sessoes:
            tr = [i for i in range(len(M)) if base[i] != teste]
            te = [i for i in reais if base[i] == teste]
            sinais_tr = {S[i] for i in tr}
            c = ClassificadorSinais(k, fator).fit([M[i] for i in tr], [S[i] for i in tr],
                                                  [G[i] for i in tr], [O[i] for i in tr])
            for i in te:
                if S[i] not in sinais_tr:
                    continue   # sinal que so essa sessao gravou: nao da para testar
                r = c.prever(M[i])
                resultados.append((S[i], r["mais_parecido"], r["conhecido"], teste))
    else:
        print("\n[AVISO] So 1 sessao: avaliando 'deixa 1 gravacao de fora'. O numero sai "
              "OTIMISTA -- grave com outra pessoa para um resultado honesto.")
        c = ClassificadorSinais(k, fator).fit(M, S, G, O)
        for i in reais:
            irmas = {j for j in range(len(M)) if O[j].split(" ")[0] == O[i].split(" ")[0]}
            if sum(1 for j in range(len(M)) if S[j] == S[i] and j not in irmas) == 0:
                continue
            r = c.prever(c.exemplos[i].matriz, _ignorar=irmas, _ja_preparada=True)
            resultados.append((S[i], r["mais_parecido"], r["conhecido"], sessoes[0]))

    if not resultados:
        print("Nada para avaliar (cada sinal precisa de 2+ gravacoes).")
        return None
    v = np.array([r[0] for r in resultados]); p = np.array([r[1] for r in resultados])
    conhecido = np.array([r[2] for r in resultados])
    acc = float((v == p).mean())
    acc_rej = float(((v == p) & conhecido).mean())
    print(f"\n=== Acerto (sinal mais parecido): {acc:.3f}  ({len(v)} testes) ===")
    print(f"    acerto E aceito pelo limiar:  {acc_rej:.3f}   "
          f"(rejeitados por estar longe demais: {(~conhecido).sum()})")
    erros = Counter((a, b) for a, b in zip(v, p) if a != b)
    if erros:
        print("Confusoes (real -> previsto):")
        for (a, b), n in erros.most_common(10):
            print(f"  {a} -> {b}: {n}")
    por_sessao = {}
    for s in sorted({r[3] for r in resultados}):
        m = np.array([r[3] == s for r in resultados])
        por_sessao[s] = float((v[m] == p[m]).mean())
    if len(por_sessao) > 1:
        print("Por sessao de teste: " + "  ".join(f"{s}: {a:.2f}" for s, a in por_sessao.items()))
    return {"acuracia": acc, "acuracia_com_limiar": acc_rej, "n_testes": int(len(v)),
            "por_sessao": por_sessao,
            "confusoes": [[a, b, int(n)] for (a, b), n in erros.most_common()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pasta", type=Path, default=SINAIS_DIR)
    p.add_argument("--saida", type=Path, default=SINAIS_MODEL_PATH)
    p.add_argument("--sem-espelho", action="store_true",
                   help="nao adiciona a copia espelhada (canhoto) de cada gravacao")
    p.add_argument("--k", type=int, default=1, help="vizinhos considerados por sinal")
    p.add_argument("--fator-limiar", type=float, default=1.3,
                   help="maior = aceita mais (rejeita menos sinais como 'desconhecido')")
    p.add_argument("--sem-avaliar", action="store_true")
    args = p.parse_args()

    gravacoes = carregar_gravacoes(args.pasta)
    M, S, G, O = montar(gravacoes, espelho=not args.sem_espelho)
    relatorio([s for s, g in zip(S, G) if not g.endswith(SUFIXO_ESPELHO)],
              [g for g in G if not g.endswith(SUFIXO_ESPELHO)])

    metricas = None if args.sem_avaliar else avaliar(M, S, G, O, args.k, args.fator_limiar)

    print("\nTreinando o modelo final com todas as gravacoes...")
    c = ClassificadorSinais(args.k, args.fator_limiar).fit(M, S, G, O)
    print(f"Limiar de 'desconhecido': {c.limiar:.3f}")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classificador": c, "versao_sinais": VERSAO_SINAIS, "sinais": c.sinais,
                 "n_gravacoes": len([g for g in G if not g.endswith(SUFIXO_ESPELHO)]),
                 "sessoes": sorted({g.removesuffix(SUFIXO_ESPELHO) for g in G}),
                 "metricas": metricas}, args.saida)
    args.saida.with_suffix(".metrics.json").write_text(
        json.dumps({"sinais": c.sinais, "limiar": c.limiar, "metricas": metricas},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo salvo em {args.saida}  ({len(c.sinais)} sinais)")


if __name__ == "__main__":
    main()

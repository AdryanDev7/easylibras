"""
Importa um dataset EXTERNO de landmarks de mao para o formato do EasyLibras.

Feito para o "Libras Landmark Dataset (A-Z)" do Kaggle (heitorccf/librasign),
mas aceita qualquer CSV de landmarks de mao do MediaPipe (21 pontos).

    # 1) baixar (precisa de: pip install kagglehub)
    python -m tools.importar_dataset --baixar

    # 2) ou apontar para a pasta/arquivo que voce baixou na mao
    python -m tools.importar_dataset --origem caminho/para/librasign

    # 3) so olhar o que tem dentro, sem gravar nada
    python -m tools.importar_dataset --origem caminho/para/librasign --inspecionar

Saida: dataset/externos/<nome>.csv, no MESMO formato do raw_landmarks.csv.
O preprocess le os dois automaticamente. Para tirar o dataset externo do
treino, e so apagar o arquivo.

Formatos de CSV reconhecidos sozinhos (o script tenta nesta ordem):

  a) colunas nomeadas por eixo+indice:  x0..x20, y0..y20, z0..z20
     (tambem x_0, X0, 0_x, x_wrist, WRIST_x, ...)
  b) 63 colunas numericas em sequencia x0,y0,z0,x1,y1,z1,...  (--ordem intercalado)
     ou x0..x20,y0..y20,z0..z20                              (--ordem blocos)
  c) 42 colunas numericas (so x,y)  -> z = 0
  d) uma pasta por letra (A/, B/, ...) com CSVs dentro -> o rotulo vem da pasta

Coluna de rotulo: label, letter, letra, class, classe, sign, sinal, target, y,
ou a unica coluna de texto. Use --col-rotulo se nao for nenhuma dessas.
Coluna de pessoa (vira "sessao"): signer, subject, person, user, participant,
pessoa, sessao, session. Se nao existir, tudo vira UMA sessao so -- leia o
GUIA.md, secao 14, para entender por que isso importa na avaliacao.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core_ai.collect_data import CABECALHO
from core_ai.config import DATASET_DIR, LETRAS_DINAMICAS

DIR_EXTERNOS = DATASET_DIR / "externos"
KAGGLE_SLUG = "heitorccf/librasign"

NOMES_ROTULO = ["label", "labels", "letter", "letra", "class", "classe", "sign",
                "sinal", "target", "y", "gesture", "char", "character"]
NOMES_PESSOA = ["signer", "signer_id", "subject", "subject_id", "person", "person_id",
                "user", "user_id", "participant", "pessoa", "sessao", "session"]
NOMES_LADO = ["handedness", "hand", "lado", "mao", "side", "hand_label"]

# Nomes oficiais dos 21 pontos no MediaPipe (alguns datasets usam isso no cabecalho)
NOMES_PONTOS_MP = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Grafias de Ç que aparecem por ai -> nome usado no config
SINONIMOS_ROTULO = {"Ç": "C_CEDILHA", "CEDILHA": "C_CEDILHA", "C_CEDILHA": "C_CEDILHA",
                    "CC": "C_CEDILHA", "Ç ": "C_CEDILHA"}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def baixar_kaggle(slug: str = KAGGLE_SLUG) -> Path:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "Para baixar automaticamente: pip install kagglehub\n"
            f"Ou baixe na mao em https://www.kaggle.com/datasets/{slug} "
            "e rode de novo com --origem <pasta>."
        )
    print(f"Baixando {slug} (fica em cache; da 2a vez e instantaneo)...")
    caminho = Path(kagglehub.dataset_download(slug))
    print(f"  -> {caminho}")
    return caminho


# --------------------------------------------------------------------------
# Leitura e deteccao de formato
# --------------------------------------------------------------------------

def listar_csvs(origem: Path) -> list[Path]:
    if origem.is_file():
        return [origem]
    arquivos = sorted(p for p in origem.rglob("*") if p.suffix.lower() in (".csv", ".tsv"))
    if not arquivos:
        outros = Counter(p.suffix.lower() for p in origem.rglob("*") if p.is_file())
        raise SystemExit(
            f"Nenhum CSV em {origem}. Extensoes encontradas: {dict(outros)}.\n"
            "Se o dataset vier em .npy/.json/.parquet, me avise que eu adapto."
        )
    return arquivos


def _ler(caminho: Path) -> pd.DataFrame:
    sep = "\t" if caminho.suffix.lower() == ".tsv" else None
    df = pd.read_csv(caminho, sep=sep, engine="python")
    # CSV sem cabecalho: o pandas usa a 1a linha de dados como nome de coluna.
    # Se quase todos os "nomes" sao numeros QUEBRADOS (0.5652...), e porque nao
    # havia cabecalho. Nomes inteiros "0,1,2,...,62" sao cabecalho de verdade
    # (e o que o pandas grava por padrao) e ficam como estao.
    numeros = [str(c) for c in df.columns if _parece_numero(c)]
    if len(numeros) >= 40 and any(not _parece_inteiro(c) for c in numeros):
        df = pd.read_csv(caminho, sep=sep, engine="python", header=None)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parece_numero(txt) -> bool:
    try:
        float(str(txt))
        return True
    except ValueError:
        return False


def _parece_inteiro(txt) -> bool:
    return str(txt).strip().lstrip("-").isdigit()


def _achar_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    por_nome = {c.lower(): c for c in df.columns}
    for nome in candidatos:
        if nome in por_nome:
            return por_nome[nome]
    return None


_RE_EIXO_IDX = [
    re.compile(r"^(?P<eixo>[xyz])_?(?P<idx>\d{1,2})$", re.I),              # x0, x_0
    re.compile(r"^(?P<idx>\d{1,2})_?(?P<eixo>[xyz])$", re.I),              # 0x, 0_x
    re.compile(r"^(?:lm|landmark|ponto|p|kp)_?(?P<idx>\d{1,2})_(?P<eixo>[xyz])$", re.I),
    re.compile(r"^(?P<eixo>[xyz])_(?:lm|landmark|ponto|p|kp)_?(?P<idx>\d{1,2})$", re.I),
]


def _mapa_colunas_nomeadas(colunas: list[str]) -> dict[tuple[int, str], str] | None:
    """Tenta achar (indice, eixo) -> coluna pelos nomes. None se nao der."""
    mapa: dict[tuple[int, str], str] = {}
    nomes_mp = {n.lower(): i for i, n in enumerate(NOMES_PONTOS_MP)}
    for col in colunas:
        c = col.strip()
        achou = None
        for rx in _RE_EIXO_IDX:
            m = rx.match(c)
            if m:
                achou = (int(m.group("idx")), m.group("eixo").lower())
                break
        if achou is None:
            m = re.match(r"^(?P<nome>[a-z_]+?)[_\.](?P<eixo>[xyz])$", c, re.I) or \
                re.match(r"^(?P<eixo>[xyz])[_\.](?P<nome>[a-z_]+)$", c, re.I)
            if m and m.group("nome").lower() in nomes_mp:
                achou = (nomes_mp[m.group("nome").lower()], m.group("eixo").lower())
        if achou and 0 <= achou[0] <= 20:
            mapa[achou] = col

    tem_xy = all((i, e) in mapa for i in range(21) for e in "xy")
    if not tem_xy:
        # aceita indices 1..21 em vez de 0..20
        if all((i, e) in mapa for i in range(1, 22) for e in "xy"):
            mapa = {(i - 1, e): c for (i, e), c in mapa.items() if 1 <= i <= 21}
            return mapa
        return None
    return mapa


def extrair_pontos(df: pd.DataFrame, ignorar: set[str], ordem: str) -> tuple[np.ndarray, str]:
    """Retorna (N,21,3) e uma descricao do formato detectado."""
    resto = [c for c in df.columns if c not in ignorar]

    mapa = _mapa_colunas_nomeadas(resto)
    if mapa is not None:
        n = len(df)
        pts = np.zeros((n, 21, 3), dtype=np.float32)
        tem_z = all((i, "z") in mapa for i in range(21))
        for i in range(21):
            pts[:, i, 0] = df[mapa[(i, "x")]].to_numpy(np.float32)
            pts[:, i, 1] = df[mapa[(i, "y")]].to_numpy(np.float32)
            if tem_z:
                pts[:, i, 2] = df[mapa[(i, "z")]].to_numpy(np.float32)
        return pts, f"colunas nomeadas ({'x,y,z' if tem_z else 'x,y sem z'})"

    numericas = [c for c in resto if pd.api.types.is_numeric_dtype(df[c])]
    # descarta colunas tipo "index"/"id" que costumam vir na frente
    numericas = [c for c in numericas
                 if c.lower() not in ("index", "id", "idx", "frame", "unnamed: 0")]
    if len(numericas) in (63, 64):
        if len(numericas) == 64:
            numericas = numericas[1:]
        M = df[numericas].to_numpy(np.float32)
        if ordem == "blocos":
            pts = np.stack([M[:, 0:21], M[:, 21:42], M[:, 42:63]], axis=-1)
        else:
            pts = M.reshape(len(M), 21, 3)
        return pts, f"63 colunas numericas, ordem '{ordem}'"
    if len(numericas) == 42:
        M = df[numericas].to_numpy(np.float32)
        if ordem == "blocos":
            xy = np.stack([M[:, 0:21], M[:, 21:42]], axis=-1)
        else:
            xy = M.reshape(len(M), 21, 2)
        pts = np.concatenate([xy, np.zeros((len(M), 21, 1), np.float32)], axis=-1)
        return pts, f"42 colunas numericas (sem z), ordem '{ordem}'"

    raise ValueError(
        f"Nao reconheci o formato: {len(numericas)} colunas numericas "
        f"(esperava 63 ou 42, ou nomes tipo x0..x20). Colunas: {list(df.columns)[:12]}..."
    )


def _rotulo_da_pasta(arquivo: Path, origem: Path) -> str | None:
    if arquivo == origem:
        return None
    pasta = arquivo.parent.name.strip().upper()
    pasta = SINONIMOS_ROTULO.get(pasta, pasta)
    if len(pasta) == 1 or pasta in SINONIMOS_ROTULO.values():
        return pasta
    return None


def normalizar_rotulo(valor) -> str:
    txt = str(valor).strip().upper()
    txt = SINONIMOS_ROTULO.get(txt, txt)
    if txt in SINONIMOS_ROTULO.values():   # "C_CEDILHA" nao pode virar "C"
        return txt
    # "letra_a", "A_1", "a.jpg" -> "A"
    m = re.fullmatch(r"(?:LETRA|LETTER|CLASS)?[_\- ]?([A-ZÇ])(?:[_\-\. ].*)?", txt)
    if m:
        return SINONIMOS_ROTULO.get(m.group(1), m.group(1))
    return txt


def detectar_espaco(pts: np.ndarray) -> str:
    """'normalizado' (0..1 do MediaPipe), 'pixels', ou 'relativo' (ja centrado)."""
    xy = pts[:, :, :2]
    lo, hi = np.nanpercentile(xy, 1), np.nanpercentile(xy, 99)
    if hi > 2.0:
        return "pixels"
    if lo < -0.05:
        return "relativo"
    return "normalizado"


# --------------------------------------------------------------------------
# Importacao
# --------------------------------------------------------------------------

def importar(origem: Path, nome: str, col_rotulo: str | None, col_pessoa: str | None,
             lado_padrao: str, largura: int, altura: int, ordem: str, espelhar: bool,
             incluir_dinamicas: bool, max_por_letra: int | None, seed: int,
             inspecionar: bool) -> Path | None:
    arquivos = listar_csvs(origem)
    print(f"{len(arquivos)} arquivo(s) CSV em {origem}")

    blocos = []
    for arq in arquivos:
        df = _ler(arq)
        c_rot = col_rotulo if col_rotulo in df.columns else _achar_coluna(df, NOMES_ROTULO)
        if c_rot is None:
            texto = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            c_rot = texto[0] if len(texto) == 1 else None
        c_pes = col_pessoa if col_pessoa in df.columns else _achar_coluna(df, NOMES_PESSOA)
        c_lado = _achar_coluna(df, NOMES_LADO)

        ignorar = {c for c in (c_rot, c_pes, c_lado) if c}
        try:
            pts, formato = extrair_pontos(df, ignorar, ordem)
        except ValueError as exc:
            print(f"  [pulado] {arq.name}: {exc}")
            continue

        if c_rot:
            rotulos = df[c_rot].map(normalizar_rotulo).to_numpy()
        else:
            r = _rotulo_da_pasta(arq, origem)
            if r is None:
                print(f"  [pulado] {arq.name}: nao achei a coluna de rotulo "
                      f"(use --col-rotulo). Colunas: {list(df.columns)[:8]}...")
                continue
            rotulos = np.array([r] * len(df))

        if c_pes:
            pessoas = df[c_pes].astype(str).str.strip().to_numpy()
        else:
            pessoas = np.array(["unica"] * len(df))

        if c_lado:
            lados = df[c_lado].astype(str).str.strip().str.lower().map(
                lambda s: "Left" if s.startswith(("l", "e")) else "Right").to_numpy()
        else:
            lados = np.array([lado_padrao] * len(df))

        print(f"  {arq.relative_to(origem) if arq != origem else arq.name}: {len(df)} linhas | "
              f"{formato} | rotulo='{c_rot or 'pasta'}' pessoa='{c_pes or '-'}' "
              f"lado='{c_lado or '-'}'")
        blocos.append((pts, rotulos, pessoas, lados))

    if not blocos:
        raise SystemExit("Nada importado.")

    pts = np.concatenate([b[0] for b in blocos])
    rotulos = np.concatenate([b[1] for b in blocos]).astype(str)
    pessoas = np.concatenate([b[2] for b in blocos]).astype(str)
    lados = np.concatenate([b[3] for b in blocos]).astype(str)

    # linhas quebradas (mao nao detectada costuma virar NaN ou tudo zero)
    validas = np.isfinite(pts).all(axis=(1, 2)) & (np.abs(pts[:, :, :2]).sum(axis=(1, 2)) > 0)
    if (~validas).any():
        print(f"\nDescartadas {(~validas).sum()} linhas sem mao (NaN/zeros).")
    pts, rotulos, pessoas, lados = pts[validas], rotulos[validas], pessoas[validas], lados[validas]

    espaco = detectar_espaco(pts)
    print(f"\nTotal: {len(rotulos)} amostras | {len(set(rotulos))} classes | "
          f"{len(set(pessoas))} pessoa(s) | lados: {dict(Counter(lados.tolist()))}")
    print(f"Coordenadas parecem estar em: {espaco}")
    print("Por letra: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(rotulos).items())))

    if inspecionar:
        print("\n(--inspecionar: nada foi gravado)")
        return None

    if not incluir_dinamicas:
        dinamicas = set(LETRAS_DINAMICAS)
        tirar = np.isin(rotulos, list(dinamicas))
        if tirar.any():
            print(f"\nRemovendo letras dinamicas {sorted(set(rotulos[tirar]))} "
                  f"({tirar.sum()} amostras). Use --incluir-dinamicas para manter.")
            pts, rotulos, pessoas, lados = pts[~tirar], rotulos[~tirar], pessoas[~tirar], lados[~tirar]

    if max_por_letra:
        rng = np.random.default_rng(seed)
        manter = []
        for letra in sorted(set(rotulos)):
            idx = np.flatnonzero(rotulos == letra)
            if len(idx) > max_por_letra:
                idx = rng.choice(idx, max_por_letra, replace=False)
            manter.extend(idx.tolist())
        manter = np.sort(np.array(manter))
        print(f"Limitando a {max_por_letra} por letra: {len(rotulos)} -> {len(manter)} amostras")
        pts, rotulos, pessoas, lados = pts[manter], rotulos[manter], pessoas[manter], lados[manter]

    # Converte para o formato do raw_landmarks.csv:
    # x,y em 0..1 relativos a img_w/img_h; preprocess multiplica de volta.
    pts = pts.copy()
    if espaco == "normalizado":
        w, h = largura, altura
    else:
        # pixels ou ja centrado: guardamos o valor como esta e usamos img 1x1,
        # assim a multiplicacao do preprocess nao altera nada.
        w, h = 1, 1
        if espaco == "relativo":
            print("[info] Coordenadas ja centradas/normalizadas pelo autor. Tudo bem: a "
                  "normalizacao do EasyLibras (pulso na origem + escala) e aplicada por cima.\n"
                  "       Mas confira se a proporcao x/y nao foi distorcida (GUIA, secao 14).")

    if espelhar:
        # Inverte SO o x, mantendo o rotulo de lado. (Inverter x E trocar o lado
        # se anula: features.py ja espelha toda mao "Left" para o referencial da
        # direita.) Na pratica isso transforma a mao no seu "reflexo" -- e o que
        # corrige um dataset gravado sem espelho e sem coluna de lado.
        if espaco == "normalizado":
            pts[:, :, 0] = 1.0 - pts[:, :, 0]
        else:
            pts[:, :, 0] = -pts[:, :, 0]
        print("[info] --espelhar: x invertido.")
        if len(set(lados)) > 1:
            print("[aviso] O dataset ja tem coluna de lado; com ela o espelho costuma ja "
                  "estar coerente. So use --espelhar se o teste mostrar que melhora.")

    DIR_EXTERNOS.mkdir(parents=True, exist_ok=True)
    saida = DIR_EXTERNOS / f"{nome}.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(CABECALHO)
        for i in range(len(rotulos)):
            sessao = f"{nome}_{pessoas[i]}"
            wr.writerow([rotulos[i], sessao, lados[i], w, h]
                        + pts[i, :, 0].tolist() + pts[i, :, 1].tolist() + pts[i, :, 2].tolist())

    print(f"\nSalvo: {saida}  ({len(rotulos)} amostras)")
    print("Proximo passo:  python -m core_ai.preprocess   e depois   python -m core_ai.train --cv")
    if len(set(pessoas)) == 1:
        print(f"\n[ATENCAO] O dataset nao identifica quem gravou: tudo virou a sessao "
              f"'{nome}_unica'. Para uma acuracia honesta, teste em sessoes gravadas por "
              f"voces:\n  python -m core_ai.train --sempre-treino '{nome}_*'")
    return saida


def main(argv=None):
    p = argparse.ArgumentParser(description="Importa dataset externo de landmarks (ex.: Kaggle librasign)")
    fonte = p.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--origem", type=Path, help="arquivo .csv ou pasta com os CSVs")
    fonte.add_argument("--baixar", nargs="?", const=KAGGLE_SLUG, metavar="SLUG",
                       help=f"baixa do Kaggle com kagglehub (padrao: {KAGGLE_SLUG})")
    p.add_argument("--nome", default="kaggle_librasign",
                   help="nome do arquivo de saida e prefixo das sessoes")
    p.add_argument("--inspecionar", action="store_true", help="so mostra o que detectou")
    p.add_argument("--col-rotulo", help="nome da coluna com a letra, se nao for detectada")
    p.add_argument("--col-pessoa", help="nome da coluna que identifica quem gravou")
    p.add_argument("--lado", default="Right", choices=["Right", "Left"],
                   help="lado assumido quando o dataset nao informa")
    p.add_argument("--resolucao", default="640x480",
                   help="resolucao das imagens originais, se as coords forem 0..1 (afeta a proporcao)")
    p.add_argument("--ordem", default="intercalado", choices=["intercalado", "blocos"],
                   help="so para CSV sem nomes de coluna: x0,y0,z0,x1... ou x0..x20,y0..y20,...")
    p.add_argument("--espelhar", action="store_true",
                   help="inverte x (use se o dataset nao tem coluna de lado e as letras "
                        "saem trocadas no teste; ver GUIA secao 14)")
    p.add_argument("--incluir-dinamicas", action="store_true",
                   help="mantem H, J, K, X, Z, Ç (por padrao saem, ver GUIA secao 8)")
    p.add_argument("--max-por-letra", type=int, default=None,
                   help="limita amostras por letra (evita que o externo engula seus dados)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    origem = baixar_kaggle(args.baixar) if args.baixar else args.origem
    if not origem.exists():
        raise SystemExit(f"Nao existe: {origem}")
    largura, altura = (int(v) for v in args.resolucao.lower().split("x"))

    importar(origem, args.nome, args.col_rotulo, args.col_pessoa, args.lado, largura, altura,
             args.ordem, args.espelhar, args.incluir_dinamicas, args.max_por_letra, args.seed,
             args.inspecionar)


if __name__ == "__main__":
    sys.exit(main())

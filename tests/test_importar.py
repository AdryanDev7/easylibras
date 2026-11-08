"""Importador de datasets externos: formatos diferentes devem dar o MESMO resultado."""
import numpy as np
import pandas as pd
import pytest

from tools import importar_dataset as imp
from tests.conftest import mao_falsa

NOMES = imp.NOMES_PONTOS_MP


@pytest.fixture
def maos():
    rng = np.random.default_rng(1)
    letras = ["A", "B", "H"] * 10          # H e dinamica: deve sair por padrao
    return letras, np.stack([mao_falsa(l, rng) for l in letras])


@pytest.fixture(autouse=True)
def saida_temporaria(tmp_path, monkeypatch):
    monkeypatch.setattr(imp, "DIR_EXTERNOS", tmp_path / "externos")


def _importar(origem, **kw):
    args = dict(nome="t", col_rotulo=None, col_pessoa=None, lado_padrao="Right",
                largura=640, altura=480, ordem="intercalado", espelhar=False,
                incluir_dinamicas=False, max_por_letra=None, seed=0, inspecionar=False)
    args.update(kw)
    return pd.read_csv(imp.importar(origem, **args))


def test_formato_nomeado_com_pessoa(tmp_path, maos):
    letras, P = maos
    df = pd.DataFrame({"letter": letras, "signer": ["s1", "s2"] * 15})
    for e, k in zip("xyz", range(3)):
        for i in range(21):
            df[f"{e}{i}"] = P[:, i, k]
    df.to_csv(tmp_path / "d.csv", index=False)
    out = _importar(tmp_path / "d.csv")
    assert set(out["label"]) == {"A", "B"}              # H removida
    assert set(out["sessao"]) == {"t_s1", "t_s2"}
    assert np.allclose(out["x3"], P[np.array(letras) != "H"][:, 3, 0], atol=1e-6)


def test_formato_sem_cabecalho_igual_ao_nomeado(tmp_path, maos):
    letras, P = maos
    M = pd.DataFrame(P.reshape(len(P), 63))
    M[63] = letras
    M.to_csv(tmp_path / "b.csv", index=False, header=False)
    out = _importar(tmp_path / "b.csv", incluir_dinamicas=True)
    assert len(out) == len(letras)                       # nenhuma linha virou cabecalho
    assert np.allclose(out[[f"y{i}" for i in range(21)]].to_numpy(), P[:, :, 1], atol=1e-6)


def test_pasta_por_letra_com_nomes_mediapipe(tmp_path, maos):
    letras, P = maos
    for letra in ("A", "B"):
        sel = P[np.array(letras) == letra]
        d = {f"{n}_{e}": sel[:, i, k] for i, n in enumerate(NOMES) for k, e in enumerate("xyz")}
        (tmp_path / "raiz" / letra).mkdir(parents=True)
        pd.DataFrame(d).to_csv(tmp_path / "raiz" / letra / "x.csv", index=False)
    out = _importar(tmp_path / "raiz")
    assert sorted(out["label"].unique()) == ["A", "B"]
    assert (out["sessao"] == "t_unica").all()


def test_descarta_linhas_sem_mao_e_limita_por_letra(tmp_path, maos):
    letras, P = maos
    P = P.copy()
    P[0] = 0.0                                          # mao nao detectada
    df = pd.DataFrame(P.reshape(len(P), 63))
    df.insert(0, "label", letras)
    df.to_csv(tmp_path / "z.csv", index=False)
    out = _importar(tmp_path / "z.csv", max_por_letra=5)
    assert out.groupby("label").size().max() == 5


def test_espelhar_inverte_so_o_x(tmp_path, maos):
    letras, P = maos
    df = pd.DataFrame(P.reshape(len(P), 63)); df.insert(0, "label", letras)
    df.to_csv(tmp_path / "e.csv", index=False)
    a = _importar(tmp_path / "e.csv")
    b = _importar(tmp_path / "e.csv", espelhar=True)
    assert np.allclose(a["x0"], 1 - b["x0"]) and np.allclose(a["y0"], b["y0"])
    assert (a["lado"] == b["lado"]).all()


@pytest.mark.parametrize("bruto,esperado", [
    ("a", "A"), (" B ", "B"), ("Ç", "C_CEDILHA"), ("letra_c", "C"), ("A_12", "A")])
def test_normaliza_rotulos(bruto, esperado):
    assert imp.normalizar_rotulo(bruto) == esperado

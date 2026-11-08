"""Sinais dinamicos: features, DTW, classificador, segmentacao e pipeline de arquivos."""
import numpy as np
import pytest

from core_ai.features_sinais import (DIM_SINAL, REFERENCIAS, Referencial, espelhar_quadros,
                                     sequencia_para_matriz)
from core_ai.sinais import ClassificadorSinais, Segmentador, dtw
from tests.sintetico_sinais import SINAIS, gerar

PESSOAS = 4


@pytest.fixture(scope="module")
def dados():
    return [(s, f"p{p}", sequencia_para_matriz(gerar(s, p, r)))
            for s in SINAIS for p in range(PESSOAS) for r in range(3)]


def test_dtw_basico():
    a = np.random.default_rng(0).normal(size=(10, 4))
    assert dtw(a, a) == pytest.approx(0.0)
    lento = np.repeat(a, 2, axis=0)                 # mesmo gesto, metade da velocidade
    outro = np.random.default_rng(1).normal(size=(10, 4))
    assert dtw(a, lento) < 0.1 * dtw(a, outro)


def test_features_formato_e_contato(dados):
    m = sequencia_para_matriz(gerar("OCULOS", 0, 0))
    assert m.shape[1] == DIM_SINAL and np.isfinite(m).all()
    olho = 35 + REFERENCIAS.index("olho")
    peito = 35 + REFERENCIAS.index("peito")
    assert m[:, olho].mean() < 0.2 < m[:, peito].mean()          # mao no olho, longe do peito
    s = sequencia_para_matriz(gerar("SAUDADE", 0, 0))
    assert s[:, peito].mean() < s[:, olho].mean()                # e o contrario na saudade


def test_invariante_a_distancia_da_camera():
    """Mesma pessoa, corpo maior/menor na imagem: features quase iguais."""
    q = gerar("TCHAU", 0, 0)
    perto = [type(x)(maos={k: v * 1.5 for k, v in x.maos.items()}, pose=x.pose * 1.5,
                     img_w=x.img_w, img_h=x.img_h, ts_ms=x.ts_ms) for x in q]
    a, b = sequencia_para_matriz(q), sequencia_para_matriz(perto)
    assert np.abs(a - b)[:, 1:7].max() < 1e-3                   # posicao relativa ao corpo


def test_reconhece_pessoa_nunca_vista(dados):
    acertos = total = 0
    for p in range(PESSOAS):
        tr = [d for d in dados if d[1] != f"p{p}"]
        te = [d for d in dados if d[1] == f"p{p}"]
        c = ClassificadorSinais().fit([d[2] for d in tr], [d[0] for d in tr], [d[1] for d in tr])
        for s, _, m in te:
            r = c.prever(m)
            acertos += r["sinal"] == s
            total += 1
    assert acertos / total >= 0.95


def test_rejeita_sinal_que_nao_conhece(dados):
    tr = [d for d in dados if d[0] != "TCHAU"]
    c = ClassificadorSinais().fit([d[2] for d in tr], [d[0] for d in tr], [d[1] for d in tr])
    aceitos = [c.prever(d[2])["conhecido"] for d in dados if d[0] == "TCHAU"]
    assert sum(aceitos) == 0


def test_classe_nada_e_ignorada(dados):
    rot = ["_NADA" if s == "TCHAU" else s for s, _, _ in dados]
    c = ClassificadorSinais().fit([d[2] for d in dados], rot, [d[1] for d in dados])
    r = c.prever(sequencia_para_matriz(gerar("TCHAU", 9, 0)))
    assert r["mais_parecido"] == "_NADA" and r["sinal"] is None
    assert "_NADA" not in c.sinais


def test_espelho_reconhece_canhoto(dados):
    M, S, G = [], [], []
    for s in SINAIS:
        for p in range(PESSOAS):
            q = gerar(s, p, 0)
            M += [sequencia_para_matriz(q), sequencia_para_matriz(espelhar_quadros(q))]
            S += [s, s]
            G += [f"p{p}", f"p{p}~espelho"]
    c = ClassificadorSinais().fit(M, S, G)
    for s in SINAIS:
        canhoto = sequencia_para_matriz(espelhar_quadros(gerar(s, 7, 1)))
        assert c.prever(canhoto)["sinal"] == s


def test_segmentador_corta_fluxo_continuo(dados):
    c = ClassificadorSinais().fit([d[2] for d in dados], [d[0] for d in dados], [d[1] for d in dados])
    fluxo, t = [], 0
    for s in ["TCHAU", "OCULOS", "NOME_IRMA"]:
        q = gerar(s, 9, 0, repouso=True)
        base = q[0].ts_ms
        for x in q:
            x.ts_ms = x.ts_ms - base + t
        t = q[-1].ts_ms + 33
        fluxo += q
    seg, saida = Segmentador(), []
    for q in fluxo:
        trecho = seg.atualizar(q, Referencial(q.pose))
        if trecho:
            saida.append(c.prever(sequencia_para_matriz(trecho))["sinal"])
    assert saida == ["TCHAU", "OCULOS", "NOME_IRMA"]


def test_pipeline_de_arquivos(tmp_path, monkeypatch):
    """coletar (salvar .npz) -> treinar_sinais -> carregar modelo -> prever."""
    import joblib
    from core_ai import coletar_sinais, treinar_sinais

    monkeypatch.setattr(coletar_sinais, "SINAIS_DIR", tmp_path / "sinais")
    for s in ["OCULOS", "SAUDADE", "TCHAU"]:
        for p in range(2):
            for r in range(2):
                coletar_sinais.salvar(gerar(s, p, r), s, f"pessoa{p}")
    saida = tmp_path / "m.joblib"
    monkeypatch.setattr("sys.argv", ["x", "--pasta", str(tmp_path / "sinais"), "--saida", str(saida)])
    treinar_sinais.main()
    pacote = joblib.load(saida)
    assert pacote["sinais"] == ["OCULOS", "SAUDADE", "TCHAU"]
    assert pacote["metricas"]["acuracia"] >= 0.9
    r = pacote["classificador"].prever(sequencia_para_matriz(gerar("SAUDADE", 5, 0)))
    assert r["sinal"] == "SAUDADE"


def test_nome_de_sinal():
    from core_ai.coletar_sinais import nome_valido
    assert nome_valido("óculos") == "OCULOS"
    assert nome_valido("nome adryan") == "NOME_ADRYAN"
    assert nome_valido("_nada") == "_NADA"

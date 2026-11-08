"""
Reconhecimento de sinais dinamicos com DTW + vizinho mais proximo.

Por que DTW e nao uma rede neural?
  - Sinal-nome e por definicao RARO: cada pessoa tem o seu, e voces vao ter
    3 a 10 gravacoes de cada um. Rede neural precisa de centenas.
  - Com DTW, ensinar um sinal novo = gravar umas 5 repeticoes. Nao tem
    "treino" pesado: as gravacoes SAO o modelo (few-shot learning).
  - O DTW (Dynamic Time Warping) alinha duas sequencias que tem a mesma forma
    mas velocidades diferentes -- uma pessoa faz OCULOS em 0,8 s, outra em 1,5 s.
  - E explicavel: da para mostrar na apresentacao "o sinal mais parecido foi a
    gravacao X do Enzo, a distancia Y".

O classificador tambem sabe dizer "NAO CONHECO" (distancia acima de um limiar
calibrado nos proprios dados), o que e essencial: sem isso, qualquer gesto
aleatorio vira algum sinal da lista.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .features_sinais import DIM_MAO, DIM_SINAL, PESOS, VERSAO_SINAIS, grupos_colunas

# desvio-padrao minimo por grupo (posicao/contato em larguras de ombro)
PISO_DESVIO = {"presente": 0.5, "posicao": 0.08, "forma": 0.05, "contato": 0.05, "entre": 0.08}


# --------------------------------------------------------------------------
# DTW
# --------------------------------------------------------------------------

def dtw(a: np.ndarray, b: np.ndarray, banda: float | None = 0.5) -> float:
    """Distancia DTW entre (n,D) e (m,D), normalizada pelo tamanho do caminho.

    banda: fracao do tamanho maximo que o alinhamento pode "desviar" da diagonal
    (Sakoe-Chiba). Evita alinhamentos absurdos e acelera.
    Implementacao vetorizada por antidiagonais: ~100 iteracoes de numpy em vez
    de n*m iteracoes de Python.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    custo = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    if banda is not None:
        r = max(int(banda * max(n, m)), abs(n - m))
        ii, jj = np.indices((n, m))
        custo = np.where(np.abs(ii * (m / n) - jj) <= r, custo, np.inf)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        D[i, j] = custo[i - 1, j - 1] + np.minimum(np.minimum(D[i - 1, j - 1], D[i - 1, j]),
                                                   D[i, j - 1])
    return float(D[n, m] / (n + m))


# --------------------------------------------------------------------------
# Classificador
# --------------------------------------------------------------------------

@dataclass
class Exemplo:
    sinal: str
    sessao: str
    origem: str          # arquivo, para mostrar "parecido com tal gravacao"
    matriz: np.ndarray   # (T, D) ja normalizada e ponderada


class ClassificadorSinais:
    def __init__(self, k: int = 1, fator_limiar: float = 1.3):
        self.k = k
        self.fator_limiar = fator_limiar
        self.media = np.zeros(DIM_SINAL, np.float32)
        self.desvio = np.ones(DIM_SINAL, np.float32)
        self.pesos = np.ones(DIM_SINAL, np.float32)
        self.exemplos: list[Exemplo] = []
        self.limiar = float("inf")
        self.versao = VERSAO_SINAIS

    # -- preparo ------------------------------------------------------------
    def _ajustar_escala(self, matrizes: list[np.ndarray]):
        tudo = np.concatenate(matrizes)
        grupos = grupos_colunas()
        self.media = tudo.mean(0)
        self.desvio = tudo.std(0)
        # As colunas de cada mao so tem sentido quando ela esta na tela. Se a
        # escala fosse calculada com os frames de mao ausente (cheios de zero),
        # o desvio ficaria minusculo -- e qualquer diferenca de posicao da mao
        # nao dominante explodiria (foi o que aconteceu nos testes).
        for mao in range(2):
            ini = mao * DIM_MAO
            presente = tudo[:, ini] > 0.5
            if presente.sum() >= 2:
                bloco = tudo[presente, ini + 1:ini + DIM_MAO]
                self.media[ini + 1:ini + DIM_MAO] = bloco.mean(0)
                self.desvio[ini + 1:ini + DIM_MAO] = bloco.std(0)
        # piso: variacao menor que isso e ruido do detector, nao informacao
        piso = np.array([PISO_DESVIO[g] for g in grupos], np.float32)
        self.desvio = np.maximum(self.desvio, piso)
        tamanho = defaultdict(int)
        for g in grupos:
            tamanho[g] += 1
        self.pesos = np.array([PESOS[g] / np.sqrt(tamanho[g] / 2) for g in grupos], np.float32)

    def preparar(self, matriz: np.ndarray) -> np.ndarray:
        return ((matriz - self.media) / self.desvio * self.pesos).astype(np.float32)

    # -- treino -------------------------------------------------------------
    def fit(self, matrizes: list[np.ndarray], sinais: list[str], sessoes: list[str],
            origens: list[str] | None = None, calibrar: bool = True):
        if not matrizes:
            raise ValueError("nenhuma gravacao")
        origens = origens or [""] * len(matrizes)
        self._ajustar_escala(matrizes)
        self.exemplos = [Exemplo(s, g, o, self.preparar(m))
                         for m, s, g, o in zip(matrizes, sinais, sessoes, origens)]
        if calibrar:
            self.limiar = self._calibrar_limiar()
        return self

    def _calibrar_limiar(self) -> float:
        """Distancia de cada gravacao ate a gravacao MAIS PARECIDA DO MESMO SINAL
        feita em OUTRA sessao (outra pessoa/dia). O limiar fica acima de quase
        todas elas: um sinal conhecido fica abaixo; um gesto fora da lista, acima.

        Por que outra sessao? Repeticoes da mesma pessoa sao quase identicas; se
        calibrar com elas, o limiar fica apertado e rejeita gente nova fazendo
        o sinal certo. So usa a mesma sessao quando o sinal so tem uma."""
        d_mesmo = []
        for i, ex in enumerate(self.exemplos):
            outros = [(o.sessao != ex.sessao, dtw(ex.matriz, o.matriz))
                      for j, o in enumerate(self.exemplos) if j != i and o.sinal == ex.sinal]
            if not outros:
                continue
            de_fora = [d for fora, d in outros if fora]
            d_mesmo.append(min(de_fora) if de_fora else min(d for _, d in outros))
        if not d_mesmo:
            return float("inf")
        return float(np.percentile(d_mesmo, 95) * self.fator_limiar)

    # -- predicao -----------------------------------------------------------
    def distancias(self, matriz_preparada: np.ndarray, ignorar: set[int] | None = None):
        """Menor distancia (media das k menores) por sinal + gravacao mais proxima."""
        por_sinal: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for j, ex in enumerate(self.exemplos):
            if ignorar and j in ignorar:
                continue
            por_sinal[ex.sinal].append((dtw(matriz_preparada, ex.matriz), j))
        resumo = {}
        for s, lst in por_sinal.items():
            lst.sort()
            resumo[s] = (float(np.mean([d for d, _ in lst[:self.k]])), lst[0][1])
        return resumo

    def prever(self, matriz: np.ndarray, top_k: int = 3, _ignorar=None, _ja_preparada=False) -> dict:
        if len(matriz) < 3:
            return {"sinal": None, "conhecido": False, "distancia": None, "confianca": 0.0,
                    "top": [], "motivo": "sequencia curta demais"}
        m = matriz if _ja_preparada else self.preparar(matriz)
        resumo = self.distancias(m, _ignorar)
        ordem = sorted(resumo.items(), key=lambda kv: kv[1][0])
        melhor, (d1, j1) = ordem[0]
        d2 = ordem[1][1][0] if len(ordem) > 1 else d1 * 2
        # confianca: quao mais perto do 1o colocado do que do 2o (0 = empate)
        confianca = float(np.clip(1.0 - d1 / max(d2, 1e-9), 0.0, 1.0))
        # sinais cujo nome comeca com "_" (ex.: _NADA) sao gestos gravados de
        # proposito para o sistema IGNORAR: coçar o rosto, ajeitar o oculos...
        conhecido = d1 <= self.limiar and not melhor.startswith("_")
        return {
            "sinal": melhor if conhecido else None,
            "conhecido": bool(conhecido),
            "mais_parecido": melhor,
            "distancia": d1,
            "limiar": self.limiar,
            "confianca": confianca,
            "gravacao_mais_proxima": self.exemplos[j1].origem,
            "top": [{"sinal": s, "distancia": d} for s, (d, _) in ordem[:top_k]],
        }

    @property
    def sinais(self) -> list[str]:
        return sorted({e.sinal for e in self.exemplos if not e.sinal.startswith("_")})


# --------------------------------------------------------------------------
# Segmentacao: quando um sinal comeca e quando termina, ao vivo
# --------------------------------------------------------------------------

class Segmentador:
    """Corta o fluxo da camera em sinais.

    Regra (a mesma que interpretes usam): entre um sinal e outro, a pessoa
    abaixa as maos ("posicao de repouso") ou tira da camera.
      - comeca: alguma mao aparece ACIMA da linha de repouso por alguns frames
      - termina: as maos ficam abaixo da linha / somem por `frames_fim` frames
    """

    def __init__(self, linha_repouso: float = 0.9, frames_inicio: int = 3,
                 frames_fim: int = 8, min_frames: int = 6, max_frames: int = 150):
        self.linha_repouso = linha_repouso   # em larguras de ombro abaixo dos ombros
        self.frames_inicio = frames_inicio
        self.frames_fim = frames_fim
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.reiniciar()

    def reiniciar(self):
        self.gravando = False
        self.buffer = []
        self._ativos = 0
        self._inativos = 0

    def ativo(self, quadro, ref) -> bool:
        for pts in quadro.maos.values():
            y_corpo = float(ref.para_corpo(pts[[0, 8, 12]])[:, 1].min())
            if y_corpo < self.linha_repouso:
                return True
        return False

    def atualizar(self, quadro, ref):
        """Recebe 1 frame. Devolve a lista de quadros do sinal quando ele termina."""
        a = self.ativo(quadro, ref)
        if not self.gravando:
            self._ativos = self._ativos + 1 if a else 0
            self.buffer = (self.buffer + [quadro])[-self.frames_inicio:]
            if self._ativos >= self.frames_inicio:
                self.gravando, self._inativos = True, 0
            return None

        self.buffer.append(quadro)
        self._inativos = 0 if a else self._inativos + 1
        acabou = self._inativos >= self.frames_fim or len(self.buffer) >= self.max_frames
        if not acabou:
            return None
        sinal = self.buffer[:len(self.buffer) - self._inativos] if self._inativos else self.buffer
        self.reiniciar()
        return sinal if len(sinal) >= self.min_frames else None

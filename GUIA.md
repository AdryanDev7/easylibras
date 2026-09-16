# EasyLibras — Guia passo a passo

Do clone até ver a letra aparecer na tela. Siga na ordem.

---

## 0. Pré-requisito que quebra tudo se estiver errado

**Use Python 3.10, 3.11 ou 3.12. Não use 3.13.**
O MediaPipe ainda não publica wheel para 3.13 — o `pip install` vai falhar ou
tentar compilar do zero por horas.

```bash
python3 --version
```

### Aviso importante sobre o MediaPipe

As versões recentes do MediaPipe **removeram o módulo `mp.solutions.hands`**.
Praticamente todo tutorial de Libras na internet usa esse módulo, e todos eles
quebram hoje com:

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

Este projeto já usa a API nova (`mediapipe.tasks` / `HandLandmarker`). Se o
código do seu colega em `src/` usa `mp.solutions.hands`, é por isso que ele
provavelmente não roda mais.

---

## 1. Clonar e preparar o ambiente

```bash
git clone https://github.com/AdryanDev7/easylibras.git
cd easylibras
git checkout -b reformulacao          # não mexe na main direto
```

### Ambiente virtual — Windows (PowerShell)

Uma linha de cada vez. O `&&` **não funciona** no PowerShell 5 (o que vem por
padrão no Windows), e `source .../bin/activate` é comando de Linux.

```powershell
py -0                                 # lista os Pythons instalados
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se aparecer erro de *"a execução de scripts foi desabilitada neste sistema"*,
libere só para esta janela e ative de novo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Deu certo quando o prompt virar `(.venv) PS C:\...`.

### Ambiente virtual — Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Instalar as dependências (igual nos dois)

```
python --version                      # tem que mostrar 3.12.x
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Windows, se der erro de DLL ao importar o MediaPipe, instale o
**Microsoft Visual C++ Redistributable 2015-2022 (x64)**.

---

## 2. Colocar os arquivos no lugar

Copie as pastas `core_ai/` e `backend_api/`, o `requirements.txt` e o
`.gitignore` para a raiz do repositório. A estrutura fica assim:

```
easylibras/
├── core_ai/                  # IA: coleta, treino, tempo real
│   ├── config.py             # caminhos, letras, limiares  <- ajuste aqui
│   ├── hand_tracker.py       # wrapper do MediaPipe (API nova)
│   ├── features.py           # normalização — o coração do projeto
│   ├── check_setup.py
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── realtime.py
├── backend_api/
│   ├── app.py                # FastAPI
│   └── services/predictor.py
├── web_demo/
│   └── index.html            # demo que roda no celular
├── dataset/                  # gerado (não versionar o CSV grande)
├── models/                   # gerado
└── requirements.txt
```

O `src/` antigo pode ficar como `src_legacy/` até vocês confirmarem que a nova
versão substitui tudo. Se o colega já coletou dados das 3 letras, **guarde o
CSV** — dá para reaproveitar (ver seção 9).

---

## 3. Testar o ambiente e a câmera

```bash
python -m core_ai.check_setup
```

Isso confere versões, baixa o modelo `hand_landmarker.task` (~7 MB, uma vez só,
vai para `models/`) e abre a webcam desenhando o esqueleto da mão. **Se este
passo funcionar, o resto funciona.** ESC para sair.

Se a câmera não abrir: `python -m core_ai.check_setup --camera 1`.

---

## 4. Coletar os dados

```bash
python -m core_ai.collect_data --letras A,B,C,D,E --sessao marcos_dia1 --amostras 300
```

Fluxo na tela: ESPAÇO começa (com contagem de 3 s), grava 300 frames com a mão
detectada, passa para a próxima letra. `N` pula uma letra, ESC aborta salvando
o que já foi gravado.

### As duas regras que definem se o projeto vai funcionar

**1. Varie durante a gravação.** Mexa a mão: mais perto, mais longe, mais alto,
mais baixo, girando um pouco o pulso. 300 frames da mão parada no mesmo lugar
valem menos que 80 frames variados — o modelo decora a posição, não o gesto.

**2. Crie uma sessão nova a cada gravação.** Pessoa diferente, dia diferente,
luz diferente, roupa diferente = `--sessao` diferente. O treino separa treino e
teste **por sessão**. Isso é o que faz a acurácia do relatório significar
alguma coisa.

> Por que isso importa: frames vizinhos de uma mesma gravação são quase
> idênticos. Se você dividir treino/teste aleatoriamente por frame, o teste
> "vaza" para dentro do treino e você recebe 99,8% de acurácia que desaba para
> 40% na hora da apresentação. Esse é o erro mais comum nesse tipo de TCC — e
> é a primeira coisa que um professor atento pergunta.

**Divisão sugerida entre os 5 integrantes:** cada um grava o alfabeto inteiro
numa sessão própria (`enzo_1`, `arthur_1`, `adryan_1`, `marcos_1`,
`fabricio_1`). Cinco mãos diferentes é o que dá generalização de verdade, e
rende um resultado honesto para o relatório.

Tudo é acrescentado em `dataset/raw_landmarks.csv`. Pode rodar quantas vezes
quiser; nada é sobrescrito.

---

## 5. Pré-processar

```bash
python -m core_ai.preprocess
```

Normaliza os pontos e imprime um relatório do dataset (quantas amostras por
letra, quantas sessões, avisos de desbalanceamento). Gera
`dataset/processed.npz`.

---

## 6. Treinar

```bash
python -m core_ai.train --cv
```

Saída: acurácia no conjunto de teste, relatório por letra, as maiores confusões
e — com `--cv`, se houver 3+ sessões — a validação cruzada por sessão.

**Use o número do GroupKFold no relatório**, não o do split simples.

Gera `models/libras_rf.joblib` e `models/libras_rf.metrics.json`.

Se aparecerem confusões altas entre duas letras específicas, são dois casos:
ou faltam dados variados daquelas duas, ou uma delas é dinâmica (seção 8).

---

## 7. Testar ao vivo

```bash
python -m core_ai.realtime
```

Mostra a letra, a barra de confiança e vai montando o texto embaixo.
Teclas: `ESC` sai, `C` limpa, `B` apaga a última letra.

A letra só é escrita quando fica estável por vários frames seguidos — sem isso
o texto vira uma sopa de letras piscando. Para ajustar a sensibilidade, mexa em
`core_ai/config.py`:

| Parâmetro | O que faz | Se estiver ruim |
|---|---|---|
| `CONF_MINIMA` | confiança mínima para aceitar | aumente se escreve letra errada |
| `FRAMES_PARA_CONFIRMAR` | frames estáveis para escrever | aumente se escreve rápido demais |
| `JANELA_SUAVIZACAO` | tamanho da votação | aumente se pisca muito |
| `SEGUNDOS_PARA_ESPACO` | tirar a mão vira espaço | ajuste ao ritmo de vocês |

---

## 8. As letras com movimento

Letras como **H, J, K, X, Z e Ç** envolvem movimento (varia um pouco por
região). Elas **não podem** ser classificadas a partir de um único frame — se
você treinar junto com as estáticas, o modelo vai confundir e derrubar a
acurácia das outras também.

Comece pelas ~21 letras estáticas (`LETRAS_ESTATICAS` em `config.py`). É um
escopo honesto, defensável na apresentação, e funciona.

Para incluir as dinâmicas depois, o caminho é: gravar janelas de ~20 frames em
vez de frames soltos, montar o vetor com os pontos normalizados **mais as
diferenças entre frames consecutivos**, e treinar um segundo classificador só
para elas. A arquitetura já está pronta para isso — `features.py` continua
sendo a mesma base.

---

## 9. Aproveitar o `data.pickle` antigo

O arquivo antigo tem **500 amostras de 5 letras: A, B, C, L e V** (100 de cada).
Formato: 42 números por amostra (x,y dos 21 pontos, intercalados), já
transladados pelo canto da caixa da mão.

Converter:

```
python -m tools.importar_pickle --arquivo data.pickle --sessao legacy_adryan
```

**Para que serve:** validar o pipeline inteiro hoje, sem câmera. Converte,
roda preprocess e train, sobe a API — se tudo isso funcionar, o único passo que
falta testar é a webcam.

**Para que NÃO serve: o modelo final.** O que o formato antigo perdeu:

| Falta | Consequência |
|---|---|
| coordenada `z` | preenchida com 0; dados novos terão z real, então não misture os dois no mesmo treino |
| qual mão foi usada | assumido `Right` (`--lado Left` se foi a esquerda) |
| resolução da câmera | assumida 640×480 (`--resolucao 1280x720` se foi outra) |
| identificação de sessão | tudo vira uma sessão só, então a acurácia sai inflada |

Recoletar as mesmas 5 letras com `collect_data.py` leva uns 5 minutos e produz
dado melhor. Vale muito mais a pena.

## 10. Subir a API

```bash
uvicorn backend_api.app:app --reload --port 8000
```

Docs interativas: http://localhost:8000/docs

| Rota | Uso |
|---|---|
| `GET /health` | status e metadados do modelo |
| `GET /labels` | letras que o modelo conhece |
| `POST /predict` | recebe os 21 pontos → retorna a letra (**recomendado**) |
| `POST /predict/frame` | recebe imagem base64 → roda MediaPipe no servidor |
| `WS /ws` | fluxo contínuo, mesmo payload do `/predict` |

**Para o app mobile, prefira `/predict`.** O celular roda o MediaPipe local e
manda ~1 KB de pontos por frame, em vez de uma imagem inteira. Fica mais rápido
e o servidor aguenta muito mais gente.

Teste rápido:

```bash
curl http://localhost:8000/health
```

---

## 11. Demo web (substitui o app mobile)

Já está pronta em `web_demo/index.html` e é servida pela própria API:

```bash
uvicorn backend_api.app:app --reload --port 8000
# abra http://localhost:8000/demo/
```

Como funciona: o **navegador** roda o MediaPipe (WASM, via CDN) e extrai os 21
pontos; manda esses pontos pela WebSocket `/ws`; o Python responde a letra. A
suavização é a mesma do `realtime.py`, então o comportamento é idêntico ao da
versão desktop. O `hand_landmarker.task` é servido pela própria API
(`/models/hand_landmarker.task`), na mesma origem — sem depender de CDN externo.

### No computador funciona direto, sem deploy nenhum

`http://localhost:8000/demo/` abre e a câmera funciona: o navegador trata
`localhost` como origem segura. **Para apresentar no notebook, é só isso.**
Nada de túnel, nada de deploy.

### Só o celular exige HTTPS

Abrindo `http://192.168.x.x:8000/demo/` no celular, a câmera não abre. Não é
bug do código: navegadores só liberam `getUserMedia` em `https://` ou
`localhost`. A página mostra um aviso explicando isso. Se você quiser mesmo
rodar no celular, três saídas:

1. **Túnel HTTPS (mais rápido para testar hoje):**
   ```bash
   cloudflared tunnel --url http://localhost:8000    # ou: ngrok http 8000
   ```
   Abre a URL `https://...` que aparece, no celular. Funciona na hora.

2. **Deploy no Railway** (a API inteira, já com a demo junto). Vira um link
   `https://` permanente — bom para a apresentação, sem depender do notebook de
   ninguém estar ligado.

3. **Flag do Chrome no celular**, só para teste: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`,
   adicione `http://192.168.x.x:8000`. Gambiarra, mas resolve numa demo rápida.

### Ajustes de desempenho

Se travar em celular fraco, em `web_demo/index.html`: troque
`delegate: "GPU"` por `"CPU"` (mais estável em alguns Androids), ou baixe a
resolução pedida no `getUserMedia` de 1280×720 para 640×480. A letra continua
saindo igual — a rede neural trabalha nos pontos, não nos pixels.

### Se depois quiserem React Native mesmo

O caminho é `react-native-vision-camera` com um frame processor chamando o
MediaPipe nativo — precisa de módulo nativo e build separado para Android e
iOS. Dá para citar no relatório como evolução, mas não é trabalho de uma
semana.

---

## 11b. Um detalhe que quebra tudo silenciosamente: o espelho

A coleta em Python faz `cv2.flip(frame, 1)` antes de detectar. A demo web
espelha o vídeo no canvas antes de chamar o `detectForVideo`. **Os dois
precisam espelhar, ou nenhum.** Se você remover de um só lado, a mão detectada
vira o inverso da que foi treinada: o MediaPipe detecta normalmente, a
confiança até sobe, e a letra sai errada — sem nenhuma mensagem de erro. Se um
dia a acurácia despencar sem motivo aparente, é aqui que se olha primeiro.

---

## 12. Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | código antigo | use os arquivos deste guia |
| `pip install mediapipe` falha | Python 3.13 | recrie o venv com 3.12 |
| `O token '&&' não é um separador válido` | PowerShell 5 | rode uma linha de cada vez |
| `source: termo não reconhecido` | comando de Linux no Windows | `.\.venv\Scripts\Activate.ps1` |
| `execução de scripts foi desabilitada` | ExecutionPolicy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `python3.12: não reconhecido` (Windows) | no Windows é o launcher `py` | `py -3.12 -m venv .venv` |
| Câmera não abre | outro app usando / índice errado | feche o outro app, tente `--camera 1` |
| `cv2.imshow` não abre janela | `opencv-python-headless` instalado | `pip uninstall opencv-python-headless` e instale `opencv-python` |
| Acurácia 99% no treino, péssima na câmera | 1 sessão só | grave 3+ sessões, com pessoas diferentes |
| Letra fica piscando na tela | suavização baixa | suba `JANELA_SUAVIZACAO` e `FRAMES_PARA_CONFIRMAR` |
| Mão detectada mas letra sempre errada | espelhamento diferente entre coleta e teste | ambos usam `cv2.flip(frame, 1)` — não remova de um só |
| `Modelo treinado com outra versao de features` | mudou `features.py` | rode `preprocess` e `train` de novo |

---

## 13. Ordem de trabalho sugerida

1. Todos rodam `check_setup` e confirmam que a câmera funciona.
2. Cada um grava sua sessão com as 5 primeiras letras. Junta os CSVs, treina,
   testa. **Só avance quando esse ciclo pequeno estiver funcionando.**
3. Aí sim gravam o alfabeto estático completo.
4. Com o modelo estável, sobe a API e abre `/demo/` no navegador.
5. Letras dinâmicas por último — e só se sobrar prazo. É um bom "trabalho
   futuro" no relatório.

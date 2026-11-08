# 🤟 EasyLibras — Tradutor de Libras em Tempo Real com IA

Aplicação desenvolvida para reconhecimento e tradução instantânea da Língua Brasileira de Sinais (Libras) para texto utilizando Visão Computacional e Inteligência Artificial.

## 👥 Integrantes da Equipe
- Enzo da Costa
- Arthur Rocha
- Adryan Lima
- Marcos Muniz
- Fabrício Lins

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Visão Computacional:** OpenCV & MediaPipe Hands
- **Machine Learning:** Scikit-Learn (Random Forest) para o alfabeto; DTW + vizinho mais próximo para sinais dinâmicos
- **Interface Mobile (Em breve):** React Native

## 🚀 Como Rodar

### 1. Preparar o ambiente
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Testar câmera e ambiente
```powershell
python -m core_ai.check_setup
```

### 3. Gravar dados
```powershell
python -m core_ai.collect_data --letras A,B,C --sessao nome_1 --amostras 300
```

### 3b. (Opcional) Usar o dataset pronto do Kaggle
```powershell
pip install kagglehub
python -m tools.importar_dataset --baixar --inspecionar
python -m tools.importar_dataset --baixar
```
Detalhes e como avaliar sem se enganar: `GUIA.md`, seção 14.

### 4. Pré-processar
```powershell
python -m core_ai.preprocess
```

### 5. Treinar

**5.1 Com cruzamento de dados**
```powershell
python -m core_ai.train --cv
```

**5.2 Sem cruzamento de dados**
```powershell
python -m core_ai.train
```

**5.3 Com dataset externo, testando só nas gravações de vocês**
```powershell
python -m core_ai.train --cv --sempre-treino "kaggle_*"
```

### 6. Testar ao vivo
```powershell
python -m core_ai.realtime
```

### 6b. Sinais de verdade (palavras e sinais-nome)
```powershell
python -m core_ai.coletar_sinais --sinais OCULOS,SAUDADE,NOME_ADRYAN --sessao adryan_1
python -m core_ai.treinar_sinais
python -m core_ai.realtime_sinais --debug
```
Duas mãos + rosto/corpo + movimento. Detalhes no `GUIA.md`, seção 15.

### 7. Subir a API + demo web
```powershell
uvicorn backend_api.app:app --reload --port 8000
```
Abra http://localhost:8000/demo/ (docs da API em `/docs`).

### 8. Testes automáticos
```powershell
python -m pytest -q
```

## 📁 Estrutura do Projeto

```text
EASYLIBRAS
│
├── core_ai/                        # Módulo de IA e Processamento
│   ├── dataset/                    # Dados brutos ou pontos extraídos (.csv / .json)
│   ├── collect_data.py             # Script para capturar gestos com MediaPipe
│   ├── preprocess.py               # Tratamento e normalização dos pontos das mãos
│   ├── train.py                    # Treinamento do modelo (TensorFlow/PyTorch)
│   └── model/                      # Modelo treinado salvo (.h5, .tflite ou .pth)
│       └── libras_model.h5
│
├── backend_api/                    # API de Tradução em Tempo Real
│   ├── app.py                      # Servidor FastAPI / Flask
│   ├── services/
│   │   ├── mediapipe_service.py    # Extração de landmarks no backend (se enviar vídeo)
│   │   └── predictor.py            # Lógica de inferência com o modelo
│   └── requirements.txt            # Dependências (opencv-python, mediapipe, tensorflow, fastapi)
│
└── mobile_app/                     # Aplicativo (Flutter ou React Native)
    ├── assets/                     # Sons, ícones e fontes
    ├── src/ / lib/
    │   ├── components/             # Botões, visualizador de câmera
    │   ├── screens/                # Tela Principal (Câmera + Tradução), Configurações
    │   └── services/               # Conexão WebSocket/HTTP com a API
    └── package.json / pubspec.yaml

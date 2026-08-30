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
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Interface Mobile (Em breve):** React Native

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

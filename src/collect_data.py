import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

DATA_DIR = './dataset'
os.makedirs(DATA_DIR, exist_ok=True)

CLASSES = ['A', 'B', 'C', 'L', 'V']
SAMPLES_PER_CLASS = 100

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
all_data = []
all_labels = []

for label in CLASSES:
    print(f"Posicione a mão para a letra: '{label}'. Aperte 'Q' na janela do vídeo para começar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Letra: {label} - Aperte 'Q' para gravar", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('EasyLibras - Coleta', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    collected = 0
    while collected < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                all_data.append(data_aux)
                all_labels.append(label)
                collected += 1

        cv2.putText(frame, f"Gravando {label}: {collected}/{SAMPLES_PER_CLASS}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('EasyLibras - Coleta', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

with open(os.path.join(DATA_DIR, 'data.pickle'), 'wb') as f:
    pickle.dump({'data': all_data, 'labels': all_labels}, f)

print(f"\nColeta concluída! Dados salvos em {DATA_DIR}/data.pickle")
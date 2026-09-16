import cv2
import mediapipe as mp
import numpy as np
import pickle

with open('./models/model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Aperte 'Q' na janela do vídeo para fechar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = prediction[0]

            x1 = max(0, int(min(x_) * W) - 10)
            y1 = max(0, int(min(y_) * H) - 10)
            x2 = min(W, int(max(x_) * W) + 10)
            y2 = min(H, int(max(y_) * H) + 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Sinal: {predicted_char}", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow('EasyLibras - Inferência em Tempo Real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
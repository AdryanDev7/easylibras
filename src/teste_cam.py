import cv2

print("Tentando abrir a webcam com DirectShow...")
# cv2.CAP_DSHOW força o Windows a usar a API de vídeo moderna sem travar
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Indice 0 falhou. Tentando indice 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERRO: Nenhuma camera foi detectada pelo OpenCV.")
else:
    print("Camera aberta com sucesso! Pressione 'q' na janela para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao receber frame.")
            break
        cv2.imshow("Teste Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
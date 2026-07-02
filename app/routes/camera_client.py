import cv2
import requests

URL = "http://127.0.0.1:8000/face/bater-ponto"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

print("Pressione ESPAÇO para bater o ponto.")
print("Pressione ESC para sair.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    cv2.imshow("Relogio de Ponto", frame)

    tecla = cv2.waitKey(1)

    if tecla == 27:  # ESC
        break

    if tecla == 32:  # ESPAÇO
        foto = "temp_webcam.jpg"
        cv2.imwrite(foto, frame)

        with open(foto, "rb") as f:
            resposta = requests.post(
                URL,
                files={"foto": f}
            )

        print(resposta.json())

cap.release()
cv2.destroyAllWindows()

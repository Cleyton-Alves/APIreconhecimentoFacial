import cv2
import json
import numpy as np
import time
import mediapipe as mp

from deepface import DeepFace
from database import SessionLocal
from models import Usuario


mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

camera = cv2.VideoCapture(0)

db = SessionLocal()

usuarios = db.query(Usuario).all()

embeddings = []

for usuario in usuarios:
    if usuario.embedding:
        embeddings.append({
            "id": usuario.id,
            "nome": usuario.nome,
            "embedding": np.array(
                json.loads(usuario.embedding)
            )
        })

print(f"{len(embeddings)} usuários carregados.")

ultimo_reconhecimento = 0
nome_detectado = "Procurando..."


while True:

    ret, frame = camera.read()

    if not ret:
        break

    rostos = []

    results = face_detection.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    if results.detections:

        h, w, _ = frame.shape

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            rostos.append((x, y, bw, bh))

    # quantidade de rostos
    cv2.putText(
        frame,
        f"Rostos: {len(rostos)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    for (x, y, w, h) in rostos:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        tempo_atual = time.time()

        if tempo_atual - ultimo_reconhecimento > 2:

            ultimo_reconhecimento = tempo_atual

            try:
                rosto = frame[y:y+h, x:x+w]

                cv2.imwrite(
                    "temp_rosto.jpg",
                    rosto
                )

                resultado = DeepFace.represent(
                    img_path="temp_rosto.jpg",
                    model_name="Facenet",
                    enforce_detection=False
                )

                embedding_novo = np.array(
                    resultado[0]["embedding"]
                )

                melhor_nome = "Desconhecido"
                menor_distancia = float("inf")

                for usuario in embeddings:

                    distancia = np.linalg.norm(
                        embedding_novo - usuario["embedding"]
                    )

                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_nome = usuario["nome"]

                if menor_distancia < 10:
                    nome_detectado = melhor_nome
                else:
                    nome_detectado = "Desconhecido"

            except Exception as e:
                print("Erro reconhecimento:", e)

        cv2.putText(
            frame,
            nome_detectado,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Reconhecimento Facial",
        frame
    )

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break


camera.release()
cv2.destroyAllWindows()
db.close()

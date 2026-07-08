import requests
import cv2
import json
import numpy as np
import time
import mediapipe as mp

from datetime import datetime
from deepface import DeepFace
from app.core.database import SessionLocal
from app.models.user import Usuario


# =========================
# CONFIG API
# =========================
API_URL = "http://127.0.0.1:8000/face/bater-ponto"


def bater_ponto(nome, tipo):
    try:
        print("CHAMOU API:", nome, tipo)

        response = requests.post(
            API_URL,
            json={
                "nome": nome,
                "tipo": tipo
            },
            timeout=5
        )

        print("STATUS:", response.status_code)
        print("RESPOSTA:", response.text)

    except Exception as e:
        print("Erro API:", e)


# =========================
# MEDIAPIPE
# =========================
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)


# =========================
# CAMERA
# =========================
camera = cv2.VideoCapture(0)


# =========================
# BANCO
# =========================
db = SessionLocal()

usuarios = db.query(Usuario).all()

embeddings = []

for usuario in usuarios:
    if usuario.embedding:
        embeddings.append({
            "id": usuario.id,
            "nome": usuario.nome,
            "embedding": np.array(json.loads(usuario.embedding))
        })

print(f"{len(embeddings)} usuários carregados.")


# =========================
# CONTROLES
# =========================
ultimo_reconhecimento = 0
INTERVALO_RECONHECIMENTO = 3

ultimo_ponto = {}
COOLDOWN = 30

estado_ponto = {}  # ENTRADA / SAIDA

nome_detectado = "Procurando..."
status_ponto = ""
tempo_status = 0
horario_ponto = ""
distancia_detectada = 0
fps = 0
tempo_fps = time.time()
contador_fps = 0
status_reconhecimento = "AGUARDANDO"

# =========================
# LOOP PRINCIPAL
# =========================
while True:

    ret, frame = camera.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    rostos = []

    results = face_detection.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            if x < 0 or y < 0 or bw <= 0 or bh <= 0:
                continue

            if x + bw > w or y + bh > h:
                continue

            rostos.append((x, y, bw, bh))

    # =========================
    # UI
    # =========================
    cv2.putText(frame, f"Rostos: {len(rostos)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    tempo_atual = time.time()
    contador_fps += 1

    if tempo_atual - tempo_fps >= 1:
        fps = contador_fps
        contador_fps = 0
        tempo_fps = tempo_atual
        
# =========================
# COR DO STATUS
# =========================
        
    if status_reconhecimento == "RECONHECIDO":
        cor_status = (0,255,0)
    elif status_reconhecimento == "DESCONHECIDO":
        cor_status = (0,0,255)
    else:
        cor_status = (0,255,255)
    
    
    # =========================
    # RECONHECIMENTO
    # =========================
    if tempo_atual - ultimo_reconhecimento > INTERVALO_RECONHECIMENTO:

        ultimo_reconhecimento = tempo_atual

        for (x, y, bw, bh) in rostos:

            try:
                rosto = frame[y:y+bh, x:x+bw]

                if rosto.size == 0:
                    continue

                cv2.imwrite("temp_rosto.jpg", rosto)

                resultado = DeepFace.represent(
                    img_path="temp_rosto.jpg",
                    model_name="Facenet",
                    enforce_detection=False
                )

                embedding_novo = np.array(resultado[0]["embedding"])

                melhor_nome = "Desconhecido"
                menor_distancia = float("inf")

                for usuario in embeddings:

                    distancia = np.linalg.norm(
                        embedding_novo - usuario["embedding"]
                    )

                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_nome = usuario["nome"]

                print("Nome:", melhor_nome)
                print("Distância:", menor_distancia)
                
                distancia_detectada = menor_distancia
                # =========================
                # PAINEL V2
                # =========================

                # Fundo do painel
                
                cv2.rectangle(frame, (5,45), (420,250), (35,35,35), -1)
                cv2.rectangle(frame, (5,45), (420,250), (70,70,70), 2)
                
                # Título
                cv2.putText(
                    frame,
                    "FACE RECOGNITION SYSTEM",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                # Usuário
                cv2.putText(
                    frame,
                    f"Usuario : {nome_detectado}",
                    (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # Distância
                cv2.putText(
                    frame,
                    f"Distancia : {distancia_detectada:.2f}",
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # Horário
                cv2.putText(
                    frame,
                    f"Horario : {horario_ponto}",
                    (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                    )
                cv2.putText(
                frame,
                f"Status : {status_reconhecimento}",
                (15,175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cor_status,
                2
                )

                cv2.putText(
                    frame,
                    f"FPS : {fps}",
                    (15,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255,255,255),
                    2
                    )

                cv2.putText(
                    frame,
                    f"Rostos : {len(rostos)}",
                    (15,225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255,255,255),
                    2
                    )


                # =========================
                # DECISÃO
                # =========================
                print("Nome:", melhor_nome)
                print("Distância:", menor_distancia)

                if menor_distancia < 2.5:

                    print(">>> RECONHECIDO")

                    nome_detectado = melhor_nome
                    status_reconhecimento = "RECONHECIDO"

                    agora = time.time()

                    ultimo = ultimo_ponto.get(melhor_nome, 0)

                    if agora - ultimo > COOLDOWN:

                        ultimo_ponto[melhor_nome] = agora

                        # alterna entrada/saida
                        estado_atual = estado_ponto.get(melhor_nome, "ENTRADA")

                        if estado_atual == "ENTRADA":
                            estado_ponto[melhor_nome] = "SAIDA"
                            tipo = "ENTRADA"
                        else:
                            estado_ponto[melhor_nome] = "ENTRADA"
                            tipo = "SAIDA"

                        bater_ponto(melhor_nome, tipo)

                        horario_ponto = datetime.now().strftime("%H:%M:%S")

                        status_ponto = "PONTO REGISTRADO"

                        tempo_status = agora
                        

                else:
                    nome_detectado = "Desconhecido"
                    status_reconhecimento = "DESCONHECIDO"

            except Exception as e:
                print("Erro reconhecimento:", e)

    # =========================
    # DESENHO
    # =========================
    for (x, y, bw, bh) in rostos:
        cor = (0, 255, 0) if nome_detectado != "Desconhecido" else (0, 0, 255)

        cv2.rectangle(
            frame,
            (x, y),
            (x + bw, y + bh),
            cor,
            2
            )

        cv2.putText(
            frame,
            nome_detectado,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor,
            2
        )

    # =========================
    # STATUS
    # =========================
    if status_ponto and time.time() - tempo_status < 3:
        cv2.putText(frame,
                    status_ponto,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
db.close()

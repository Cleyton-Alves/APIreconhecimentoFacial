from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session

import cv2
import json
import numpy as np
import os
import shutil

from deepface import DeepFace

from database import get_db
from models import Usuario, RegistroPonto
from schemas import UsuarioResponse, RegistroPontoDetalhado
from typing import List

os.makedirs("faces", exist_ok=True)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API Facial Online"}


@app.get("/usuarios", response_model=List[UsuarioResponse])
def listar_usuarios(
    db: Session = Depends(get_db)
):
    return db.query(Usuario).all()


@app.get(
    "/pontos-detalhado",
    response_model=list[RegistroPontoDetalhado]
)
def listar_pontos_detalhado(
    db: Session = Depends(get_db)
):
    return db.query(RegistroPonto).all()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    caminho = f"faces/{file.filename}"

    with open(caminho, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    imagem = cv2.imread(caminho)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    rostos = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return {
        "arquivo": file.filename,
        "rostos_encontrados": len(rostos)
    }


@app.post("/comparar")
async def comparar(
    foto1: UploadFile = File(...),
    foto2: UploadFile = File(...)
):

    caminho1 = f"temp_{foto1.filename}"
    caminho2 = f"temp_{foto2.filename}"

    with open(caminho1, "wb") as buffer:
        shutil.copyfileobj(foto1.file, buffer)

    with open(caminho2, "wb") as buffer:
        shutil.copyfileobj(foto2.file, buffer)

    try:
        resultado = DeepFace.verify(
            img1_path=caminho1,
            img2_path=caminho2,
            enforce_detection=True
        )

        return {
            "mesma_pessoa": resultado["verified"],
            "distancia": resultado["distance"]
        }

    finally:
        if os.path.exists(caminho1):
            os.remove(caminho1)

        if os.path.exists(caminho2):
            os.remove(caminho2)

@app.post("/cadastrar")
async def cadastrar(
    nome: str,
    foto: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    extensao = foto.filename.split(".")[-1]
    caminho = f"faces/{nome}.{extensao}"

    with open(caminho, "wb") as buffer:
        shutil.copyfileobj(foto.file, buffer)

    try:

        embedding = DeepFace.represent(
            img_path=caminho,
            model_name="Facenet",
            enforce_detection=False
        )

        embedding_json = json.dumps(
            embedding[0]["embedding"]
        )

    except Exception as e:

        return {
            "erro": f"Falha ao gerar embedding: {str(e)}"
        }

    usuario = Usuario(
        nome=nome,
        foto=caminho,
        embedding=embedding_json
    )

    db.add(usuario)
    db.commit()
    db.refresh(usuario)

    return {
        "mensagem": "Usuário cadastrado",
        "id": usuario.id,
        "nome": usuario.nome
    }


@app.post("/reconhecer")
async def reconhecer(
    foto: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    caminho_temp = f"temp_{foto.filename}"

    with open(caminho_temp, "wb") as buffer:
        shutil.copyfileobj(foto.file, buffer)

    try:

        resultado = DeepFace.represent(
            img_path=caminho_temp,
            model_name="Facenet",
            enforce_detection=False
        )

        embedding_novo = np.array(resultado[0]["embedding"])

        usuarios = db.query(Usuario).all()

        melhor_usuario = None
        menor_distancia = float("inf")

        for usuario in usuarios:

            if not usuario.embedding:
                continue

            embedding_salvo = np.array(
                json.loads(usuario.embedding)
            )

            distancia = np.linalg.norm(
                embedding_novo - embedding_salvo
            )

            if distancia < menor_distancia:
                menor_distancia = distancia
                melhor_usuario = usuario

        if melhor_usuario and menor_distancia < 10:

            return {
                "encontrado": True,
                "id": melhor_usuario.id,
                "nome": melhor_usuario.nome,
                "distancia": float(menor_distancia)
            }

        return {
            "encontrado": False,
            "mensagem": "Nenhum rosto compatível"
        }

    finally:

        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)
            
@app.post("/bater-ponto")
async def bater_ponto(
    foto: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    caminho_temp = f"temp_{foto.filename}"

    with open(caminho_temp, "wb") as buffer:
        shutil.copyfileobj(foto.file, buffer)

    try:

        for arquivo in os.listdir("faces"):

            caminho_cadastrado = os.path.join("faces", arquivo)

            resultado = DeepFace.verify(
                img1_path=caminho_temp,
                img2_path=caminho_cadastrado,
                enforce_detection=False
            )

            if resultado["verified"]:

                nome_arquivo = os.path.splitext(arquivo)[0]

                usuario = db.query(Usuario).filter(
                    Usuario.nome == nome_arquivo
                ).first()

                if usuario:

                    registro = RegistroPonto(
                        usuario_id=usuario.id
                    )

                    db.add(registro)
                    db.commit()
                    db.refresh(registro)

                    return {
                        "ponto_registrado": True,
                        "usuario": usuario.nome,
                        "horario": registro.data_hora
                    }

        return {
            "ponto_registrado": False,
            "mensagem": "Usuário não reconhecido"
        }

    finally:

        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)


@app.get("/pontos")
def listar_pontos(
    db: Session = Depends(get_db)
):

    registros = db.query(RegistroPonto).all()

    resultado = []

    for registro in registros:

        usuario = db.query(Usuario).filter(
            Usuario.id == registro.usuario_id
        ).first()

        resultado.append({
            "id": registro.id,
            "usuario": usuario.nome,
            "data_hora": registro.data_hora
        })

    return resultado
            
from fastapi import APIRouter, UploadFile, File
from datetime import datetime
import os
from deepface import DeepFace

from app.core.database import SessionLocal
from app.models.user import Usuario, RegistroPonto

router = APIRouter()

@router.post("/opcao3/reconhecer-e-bater-ponto")
async def reconhecer_e_bater_ponto(foto: UploadFile = File(...)):
    db = SessionLocal()

    caminho_temp = f"temp_{foto.filename}"

    with open(caminho_temp, "wb") as buffer:
        buffer.write(await foto.read())

    try:
        usuarios = db.query(Usuario).all()

        for usuario in usuarios:
            try:
                resultado = DeepFace.verify(
                    img1_path=caminho_temp,
                    img2_path=usuario.foto,
                    enforce_detection=False
                )

                if resultado["verified"]:
                    ponto = RegistroPonto(
                        usuario_id=usuario.id,
                        data_hora=datetime.now()
                    )

                    db.add(ponto)
                    db.commit()

                    os.remove(caminho_temp)

                    return {
                        "status": "ok",
                        "usuario": usuario.nome,
                        "mensagem": "Ponto registrado"
                    }

            except Exception:
                continue

        os.remove(caminho_temp)

        return {
            "status": "not_found",
            "mensagem": "Usuário não reconhecido"
        }

    finally:
        db.close()
        
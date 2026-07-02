from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.user import Usuario, RegistroPonto
from app.schemas.user import UsuarioResponse, RegistroPontoDetalhado

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/usuarios", response_model=List[UsuarioResponse])
def listar_usuarios(db: Session = Depends(get_db)):
    return db.query(Usuario).all()


@router.get("/pontos-detalhado", response_model=List[RegistroPontoDetalhado])
def listar_pontos_detalhado(db: Session = Depends(get_db)):
    return db.query(RegistroPonto).all()


@router.get("/pontos")
def listar_pontos(db: Session = Depends(get_db)):
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

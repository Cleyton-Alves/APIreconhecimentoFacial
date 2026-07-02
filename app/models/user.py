from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String(100), unique=True, nullable=False)
    foto = Column(String(255), nullable=False)
    embedding = Column(Text, nullable=True)
    data_cadastro = Column(DateTime, default=datetime.now)

    registros_ponto = relationship(
        "RegistroPonto",
        back_populates="usuario"
    )


class RegistroPonto(Base):
    __tablename__ = "registros_ponto"

    id = Column(Integer, primary_key=True, index=True)

    usuario_id = Column(
        Integer,
        ForeignKey("usuarios.id"),
        nullable=False
    )

    data_hora = Column(
        DateTime,
        default=datetime.now
    )

    usuario = relationship(
        "Usuario",
        back_populates="registros_ponto"
    )
    
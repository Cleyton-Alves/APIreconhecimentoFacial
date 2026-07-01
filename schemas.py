from pydantic import BaseModel
from datetime import datetime

class UsuarioResponse(BaseModel):
    id: int
    nome: str
    foto: str

    class Config:
        from_attributes = True


class RegistroPontoResponse(BaseModel):
    id: int
    usuario_id: int
    data_hora: datetime

    class Config:
        from_attributes = True


class RegistroPontoDetalhado(BaseModel):
    id: int
    data_hora: datetime
    usuario: UsuarioResponse

    class Config:
        from_attributes = True
        
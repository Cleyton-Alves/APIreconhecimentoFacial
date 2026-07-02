from fastapi import FastAPI
from app.routes import user_router, face_router
from app.routes import reconhecimento
app = FastAPI(title="Face Recognition API")

app.include_router(user_router.router)
app.include_router(face_router.router)
app.include_router(reconhecimento.router)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import user_router, face_router
from app.routes import reconhecimento

app = FastAPI(title="Face Recognition API")

# 🔥 CORS (ESSENCIAL PARA O DASHBOARD)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # libera dashboard (5500, file, etc)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(user_router.router)
app.include_router(face_router.router)
app.include_router(reconhecimento.router)

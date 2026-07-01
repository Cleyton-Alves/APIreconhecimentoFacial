from deepface import DeepFace

resultado = DeepFace.verify(
    img1_path="faces/foto1.jpg",
    img2_path="faces/foto2.jpg"
)

print(resultado)

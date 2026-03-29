from fastapi import FastAPI, UploadFile, File
import shutil
import tensorflow as tf
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224
model = tf.keras.models.load_model("best_model.h5")

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred < 0.5:
        return {"prediction": "Caries", "confidence": float(1 - pred)}
    else:
        return {"prediction": "No Caries", "confidence": float(pred)}

@app.get("/")
async def home():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = "temp.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)
    return result
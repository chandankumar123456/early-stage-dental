import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224

model = tf.keras.models.load_model("best_model.h5")

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return {"prediction": "Caries", "confidence": float(pred)}
    else:
        return {"prediction": "No Caries", "confidence": float(1 - pred)}

print(predict("test.jpg"))
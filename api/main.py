from fastapi import FastAPI,File,UploadFile
from numpy.lib.type_check import imag
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/")
@app.get("/ping")
async def ping():
    return "My fast api server is working fine!!!"


def file_to_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
    image = file_to_image(await file.read())
    batch_image = np.expand_dims(image,axis=0)
    #MODEL.predict(batch_image)





if __name__ == "__main__":
    uvicorn.run(app , host="localhost" , port=8000)

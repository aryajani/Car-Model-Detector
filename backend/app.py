from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv
import os
import io

print("going to load_dotenv")
load_dotenv()
print("GROQ KEY:", os.getenv("GROQ_API_KEY"))
app = FastAPI()

from load_model import load_model
from predict_car import predict
from chatbot import get_car_info



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, classes = load_model()

@app.post("/predict")
async def classify_car(file: UploadFile):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    class_index = predict(model, image)
    class_name = classes[class_index]

    return {
        "class_id": class_index,
        "class_name": class_name
    }

@app.get("/car-info")
def car_info(name: str):

    info = get_car_info(name)

    return {"info": info}
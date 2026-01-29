from fastapi import FastAPI
from emotion_detection.service.api.api import main_router
import onnxruntime as rt
import os

app = FastAPI(project_name="Emotions Detection")
app.include_router(main_router)

providers = ['CPUExecutionProvider']

MODEL_PATH = os.path.join(os.path.dirname(__file__), "vit_classifier.onnx")

m_q = rt.InferenceSession(
    MODEL_PATH,
    providers=providers
)

@app.get("/")
async def root():
    return {"hello": "world"}

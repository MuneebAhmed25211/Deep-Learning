from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
emo_router = APIRouter()
from emotion_detection.service.core.logic.onnx_inference import emotion_detector
from emotion_detection.service.core.schemas.output import APIOutput

@emo_router.post("/detect", response_model=APIOutput)
async def detect(im:UploadFile):

    if im.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(
            status_code=415, detail="Not an image"
        )
    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)

    return emotion_detector(image)
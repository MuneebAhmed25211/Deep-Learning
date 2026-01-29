import onnxruntime as rt
import cv2
import numpy as np
from emotion_detection.service.core.logic.model_loader import m_q
def emotion_detector(img_array):



    # Resize
    test_image = cv2.resize(img_array, (224, 224))

    # Convert to float + normalize (recommended)
    test_image = test_image.astype(np.float32) / 255.0

    # 🔥 Convert HWC -> CHW
    test_image = np.transpose(test_image, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(test_image, axis=0)

    print(input_tensor.shape)   # should be (1, 3, 224, 224)

    onnx_pred = m_q.run(None, {"input": input_tensor})

    pred = np.argmax(onnx_pred[0][0])

    if pred == 0:
        emotion = "angry"
    elif pred == 1:
        emotion = "happy"
    else:
        emotion = "sad"

    return {"emotion": emotion}

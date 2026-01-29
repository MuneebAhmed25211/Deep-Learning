import onnxruntime as rt

providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession(
        "vit_classifier.onnx",
        providers=providers
    )
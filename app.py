from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Load YOLOv8 ONNX model
session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO classes (important ones)
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"count": 0, "objects": [], "description": []})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # Inference
    outputs = session.run([output_name], {input_name: img_input})[0][0]

    detections = []
    descriptions = []

    for det in outputs:
        conf = det[4]
        if conf > 0.4:
            cls = int(np.argmax(det[5:]))
            label = CLASSES[cls]

            x, y, bw, bh = det[:4]
            x1 = int((x - bw / 2) * w / 640)
            y1 = int((y - bh / 2) * h / 640)
            x2 = int((x + bw / 2) * w / 640)
            y2 = int((y + bh / 2) * h / 640)

            detections.append({
                "label": label,
                "confidence": float(conf),
                "box": [x1, y1, x2, y2]
            })
            descriptions.append(label)

    return jsonify({
        "count": len(detections),
        "objects": detections,
        "description": list(set(descriptions))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

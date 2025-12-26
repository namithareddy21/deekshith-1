from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    output = session.run([output_name], {input_name: img_input})[0]
    preds = np.squeeze(output).T   # ✅ CRITICAL FIX

    boxes = []
    labels = []

    for pred in preds:
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.4:
            cx, cy, bw, bh = pred[:4]

            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)

            label = CLASSES[class_id]

            boxes.append({
                "label": label,
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2]
            })
            labels.append(label)

    return jsonify({
        "count": len(boxes),
        "objects": boxes,
        "description": list(set(labels))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
from collections import Counter

app = Flask(__name__)

# ---------------------------
# Load YOLOv8 ONNX model
# ---------------------------
MODEL_PATH = "yolov8n.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO class names
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ---------------------------
# NMS
# ---------------------------
def nms(boxes, scores, score_thresh=0.25, iou_thresh=0.45):
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_thresh, iou_thresh
    )
    return indices

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"objects": [], "counts": {}})

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None]

    # Inference
    outputs = session.run([output_name], {input_name: img_input})[0]
    preds = np.squeeze(outputs).T  # (8400, 85)

    boxes = []
    confidences = []
    class_ids = []

    for pred in preds:
        obj_conf = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        confidence = obj_conf * class_conf

        if confidence > 0.25:
            cx, cy, bw, bh = pred[:4]

            x = int((cx - bw / 2) * w / 640)
            y = int((cy - bh / 2) * h / 640)
            bw = int(bw * w / 640)
            bh = int(bh * h / 640)

            boxes.append([x, y, bw, bh])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = nms(boxes, confidences)

    objects = []
    labels = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = CLASSES[class_ids[i]]

            objects.append({
                "label": label,
                "confidence": round(confidences[i], 2),
                "box": [x, y, x + bw, y + bh],
                "width_px": bw,
                "height_px": bh
            })
            labels.append(label)

    counts = dict(Counter(labels))

    return jsonify({
        "objects": objects,
        "counts": counts
    })

# ---------------------------
# Run (Render compatible)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

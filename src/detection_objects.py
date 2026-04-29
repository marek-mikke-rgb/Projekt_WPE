import cv2
import numpy as np
import os

# Automatyczne wykrycie katalogu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_yolo():
    coco_path = os.path.join(MODEL_DIR, "coco.names")
    cfg_path = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
    weights_path = os.path.join(MODEL_DIR, "yolov3-tiny.weights")

    # Wczytanie klas COCO
    with open(coco_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Wczytanie YOLO
    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes


def detect_objects(frame, net, output_layers, classes, conf_threshold=0.5):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    detections = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                cx, cy, w, h = detection[:4]
                x = int(cx * width - w * width / 2)
                y = int(cy * height - h * height / 2)
                detections.append((x, y, int(w * width), int(h * height), class_id, confidence))

    return detections

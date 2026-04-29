import cv2
from detection_objects import load_yolo, detect_objects

def main():
    net, output_layers, classes = load_yolo()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame, net, output_layers, classes)

        for (x, y, w, h, class_id, conf) in detections:
            label = f"{classes[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detekcja obiektow (YOLO)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

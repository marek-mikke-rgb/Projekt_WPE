import cv2

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

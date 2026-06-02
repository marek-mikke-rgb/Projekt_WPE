import cv2
import numpy as np
from scipy.fft import fft, fftfreq


def handle_mouse(event, x, y, flags, param):
    """Niestandardowy kontroler myszy do obsługi pionowej manetki"""
    system = param
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        t_x, t_y, t_w, t_h = 70, 380, 70, 300
        hit_padding = 30

        if (t_x - hit_padding) <= x <= (t_x + t_w + hit_padding) and (t_y - hit_padding) <= y <= (
                t_y + t_h + hit_padding):
            val = 1.0 - (y - t_y) / float(t_h)
            system.throttle = max(0.0, min(1.0, val))


class MarineTerminal:
    def __init__(self, fps=30.0, width=1920, height=1080):
        self.W = width
        self.H = height
        self.vid_w = 1280
        self.vid_h = 720

        self.throttle = 0.0
        self.speed_boat = 0.0
        self.alpha = 0.05

        self.fps = fps
        self.sift = cv2.SIFT_create()
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.max_history_frames = int(self.fps * 10)
        self.brightness_history = []

        self.spectrum_x = []
        self.spectrum_y = []
        self.current_f = 0.0
        self.current_T = 0.0
        self.current_L = 0.0
        self.speed_px_s = 0.0

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def extract_initial_features(self, gray_frame):
        keypoints = self.sift.detect(gray_frame, None)
        if not keypoints:
            return None
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:100]
        return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    def update_wave_parameters(self, gray_frame):
        h, w = gray_frame.shape
        center_patch = gray_frame[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
        self.brightness_history.append(np.mean(center_patch))

        if len(self.brightness_history) > self.max_history_frames:
            self.brightness_history.pop(0)

        if len(self.brightness_history) == self.max_history_frames:
            data = np.array(self.brightness_history)
            data = data - np.mean(data)

            yf = fft(data)
            xf = fftfreq(self.max_history_frames, 1 / self.fps)

            positive_yf = np.abs(yf[1:self.max_history_frames // 2])
            positive_xf = xf[1:self.max_history_frames // 2]

            if len(positive_yf) > 0:
                self.spectrum_x = positive_xf
                self.spectrum_y = positive_yf

                peak_idx = np.argmax(positive_yf)
                f = positive_xf[peak_idx]

                if 0.2 < f < 10.0:
                    self.current_f = f
                    self.current_T = 1.0 / f
                    L_deep_water = (9.81 * (self.current_T ** 2)) / (2.0 * np.pi)
                    calibration_factor = 0.9404
                    self.current_L = L_deep_water * calibration_factor

    def update_boat_physics(self):
        target_speed = self.throttle * 40.0
        self.speed_boat += self.alpha * (target_speed - self.speed_boat)

    def render_ui(self, clean_video_frame):
        canvas = np.full((self.H, self.W, 3), 20, dtype=np.uint8)

        # --- DYNAMICZNE PROPORCJE I WYRÓWNANIE DO PRAWEJ ---
        orig_h, orig_w = clean_video_frame.shape[:2]
        aspect_ratio = orig_w / orig_h

        # Utrzymujemy stałą wysokość 720px, wyliczamy szerokość
        new_h = self.vid_h
        new_w = int(new_h * aspect_ratio)

        # Wyrównanie do prawej: prawy brzeg wideo kończy się tam, gdzie ramka FFT (1840px)
        v_x = 1840 - new_w
        v_y = 40

        # Zabezpieczenie: jeśli wideo ma ekstremalne proporcje (np. ultrapanorama),
        # wymuszamy, aby nie weszło na lewy panel (który kończy się na 520px + margines)
        if v_x < 560:
            v_x = 560
            new_w = 1840 - 560

        resized_video = cv2.resize(clean_video_frame, (new_w, new_h))
        canvas[v_y:v_y + new_h, v_x:v_x + new_w] = resized_video

        border_color = (0, 200, 255)
        cv2.rectangle(canvas, (v_x, v_y), (v_x + new_w, v_y + new_h), border_color, 2)

        # --- PANELE TŁA ---
        cv2.rectangle(canvas, (40, 40), (520, 760), (30, 30, 30), -1)
        cv2.rectangle(canvas, (40, 40), (520, 760), border_color, 2)
        cv2.rectangle(canvas, (40, 800), (1840, 1040), (30, 30, 30), -1)
        cv2.rectangle(canvas, (40, 800), (1840, 1040), border_color, 2)

        font = cv2.FONT_HERSHEY_DUPLEX

        # --- DANE FIZYCZNE ---
        cv2.putText(canvas, "MARINE TERMINAL", (60, 90), font, 1.0, (0, 255, 0), 2)
        cv2.line(canvas, (60, 110), (500, 110), border_color, 1)

        cv2.putText(canvas, f"PREDK. WODY: {self.speed_px_s:.1f} px/s", (60, 160), font, 0.7, (0, 255, 0), 1)
        cv2.putText(canvas, f"CZESTOTLIWOSC: {self.current_f:.2f} Hz", (60, 210), font, 0.7, (255, 255, 255), 1)
        cv2.putText(canvas, f"OKRES FALI (T): {self.current_T:.2f} s", (60, 260), font, 0.7, (255, 255, 255), 1)
        cv2.putText(canvas, f"DLUGOSC (L): {self.current_L:.2f} m", (60, 310), font, 0.7, border_color, 1)

        cv2.line(canvas, (60, 340), (500, 340), (60, 60, 60), 1)

        # --- ELEGANCKA PIONOWA MANETKA ---
        t_x, t_y, t_w, t_h = 70, 380, 70, 300
        cv2.rectangle(canvas, (t_x, t_y), (t_x + t_w, t_y + t_h), (15, 15, 15), -1)
        cv2.rectangle(canvas, (t_x, t_y), (t_x + t_w, t_y + t_h), (80, 80, 80), 2)

        fill_h = int(self.throttle * t_h)
        fill_y = t_y + t_h - fill_h
        cv2.rectangle(canvas, (t_x, fill_y), (t_x + t_w, t_y + t_h), (0, 150, 255), -1)

        h_y = fill_y
        cv2.rectangle(canvas, (t_x - 15, h_y - 12), (t_x + t_w + 15, h_y + 12), (200, 200, 200), -1)
        cv2.rectangle(canvas, (t_x - 15, h_y - 12), (t_x + t_w + 15, h_y + 12), (255, 255, 255), 2)
        cv2.line(canvas, (t_x - 5, h_y), (t_x + t_w + 5, h_y), (100, 100, 100), 2)

        cv2.putText(canvas, "MOC", (t_x + 10, t_y - 15), font, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, f"{int(self.throttle * 100)}%", (t_x + 10, t_y + t_h + 30), font, 0.7, (0, 150, 255), 2)

        # --- PRĘDKOŚĆ ŁODZI ---
        cv2.putText(canvas, "PREDKOSC LODZI:", (180, 480), font, 0.7, (200, 200, 200), 1)
        cv2.putText(canvas, f"{self.speed_boat:.1f} kn", (380, 475), font, 0.8, (0, 255, 0), 2)
        cv2.rectangle(canvas, (180, 500), (480, 520), (50, 50, 50), -1)
        speed_ratio = min(self.speed_boat / 40.0, 1.0)
        cv2.rectangle(canvas, (180, 500), (180 + int(speed_ratio * 300), 520), (0, 255, 0), -1)

        # --- DOLNY PANEL: WIDMO FFT Ze SKALĄ X ---
        cv2.putText(canvas, "WIDMO CZESTOTLIWOSCI (FFT)", (60, 840), font, 0.8, (0, 255, 0), 1)

        base_y = 1000
        cv2.line(canvas, (60, base_y), (1820, base_y), (150, 150, 150), 2)
        cv2.putText(canvas, "Hz", (1800, base_y + 25), font, 0.6, (150, 150, 150), 1)

        if len(self.spectrum_y) > 0 and len(self.spectrum_x) > 0:
            bar_width = 10
            spacing = 2
            max_h = 130

            max_val = np.max(self.spectrum_y)
            scale_factor = (max_h / max_val) if max_val > 0 else 1.0

            for i, val in enumerate(self.spectrum_y):
                if i >= 140: break

                h = int(val * scale_factor)
                color = (0, 255 - int((i / 140) * 100), 100 + int((i / 140) * 155))
                center_x = 60 + i * (bar_width + spacing)
                cv2.rectangle(canvas, (center_x, base_y - h), (center_x + bar_width, base_y), color, -1)

            max_freq = int(np.ceil(np.max(self.spectrum_x)))
            for hz in range(1, max_freq + 1):
                idx = np.argmin(np.abs(self.spectrum_x - hz))

                if idx < 140:
                    tick_x = 60 + idx * (bar_width + spacing) + bar_width // 2
                    cv2.line(canvas, (tick_x, base_y), (tick_x, base_y + 8), (200, 200, 200), 2)
                    cv2.putText(canvas, f"{hz}", (tick_x - 5, base_y + 25), font, 0.5, (200, 200, 200), 1)

        return canvas


def main():
    window_name = 'System wizyjny'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    SOURCE = "yt1.mp4"
    # SOURCE = 0
    cap = cv2.VideoCapture(SOURCE)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0 or np.isnan(actual_fps): actual_fps = 30.0

    system = MarineTerminal(fps=actual_fps)
    cv2.setMouseCallback(window_name, handle_mouse, system)

    ret, old_frame = cap.read()
    if not ret: return

    old_gray = system.preprocess(old_frame)
    p0 = system.extract_initial_features(old_gray)

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        system.update_boat_physics()

        frame_gray = system.preprocess(frame)
        system.update_wave_parameters(frame_gray)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **system.lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                total_distance = 0.0
                valid_points = len(good_new)

                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    total_distance += np.sqrt((a - c) ** 2 + (b - d) ** 2)

                avg_pixel_speed = (total_distance / valid_points) if valid_points > 0 else 0
                raw_speed = avg_pixel_speed * actual_fps

                # Uśrednianie wykładnicze (0.1 to współczynnik gładkości)
                system.speed_px_s += 0.1 * (raw_speed - system.speed_px_s)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = system.extract_initial_features(frame_gray)

        ui_canvas = system.render_ui(frame)
        cv2.imshow(window_name, ui_canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            p0 = system.extract_initial_features(frame_gray)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
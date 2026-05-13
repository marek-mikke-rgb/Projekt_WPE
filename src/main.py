import cv2
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class WaveAnalyzer:
    def __init__(self, focal_length=3.04, pixel_size=0.00112, width=1920, height=1080, fps=30.0):
        self.f0 = focal_length
        self.d_u = pixel_size
        self.d_v = pixel_size
        self.u_h = width / 2.0
        self.v_h = height / 2.0
        self.fps = fps

        self.sift = cv2.SIFT_create()
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.color = np.random.randint(0, 255, (1000, 3))

        # --- Bufor do analizy czasowej (FFT) ---
        # Zwiększono do 10 sekund dla uśrednienia większej liczby próbek
        self.max_history_frames = int(fps * 10)
        self.brightness_history = []

        # Zmienne do wykresu FFT
        self.spectrum_x = []
        self.spectrum_y = []

        # Zmienne numeryczne
        self.current_f = 0.0
        self.current_T = 0.0
        self.current_L = 0.0

        # Inicjalizacja Matplotlib (bez interaktywnego okna)
        self.fig, self.ax = plt.subplots(figsize=(4, 4.8), dpi=100)
        self.canvas = FigureCanvas(self.fig)

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
        """Analiza jasności w czasie i FFT"""
        h, w = gray_frame.shape
        # Analiza jasności ze środkowego wycinka obrazu
        center_patch = gray_frame[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
        mean_brightness = np.mean(center_patch)

        self.brightness_history.append(mean_brightness)

        if len(self.brightness_history) > self.max_history_frames:
            self.brightness_history.pop(0)

        if len(self.brightness_history) == self.max_history_frames:
            data = np.array(self.brightness_history)
            data = data - np.mean(data)  # Normalizacja (odjęcie średniej DC)

            yf = fft(data)
            xf = fftfreq(self.max_history_frames, 1 / self.fps)

            # Bierzemy tylko dodatnią połowę widma
            positive_yf = np.abs(yf[1:self.max_history_frames // 2])
            positive_xf = xf[1:self.max_history_frames // 2]

            if len(positive_yf) > 0:
                self.spectrum_x = positive_xf
                self.spectrum_y = positive_yf

                peak_idx = np.argmax(positive_yf)
                f = positive_xf[peak_idx]

                # Odfiltrowanie szumów i błędnych odczytów
                if 0.2 < f < 10.0:
                    self.current_f = f
                    self.current_T = 1.0 / f
                    self.current_L = (9.81 * (self.current_T ** 2)) / (2.0 * np.pi)

    def get_spectrum_image(self):
        """Renderuje aktualny wykres FFT w formie słupkowej"""
        if len(self.spectrum_x) > 0:
            self.ax.clear()  # Czyścimy wykres, by narysować klatkę na nowo

            self.ax.set_title("Widmo Amplitudowe FFT")
            self.ax.set_xlabel("Czestotliwosc [Hz]")
            self.ax.set_ylabel("Amplituda")

            # Obliczenie idealnej szerokości słupka dla aktualnej rozdzielczości FFT
            bar_width = self.spectrum_x[1] - self.spectrum_x[0]

            # Rysowanie wykresu słupkowego
            self.ax.bar(self.spectrum_x, self.spectrum_y, width=bar_width * 0.8, color='dodgerblue')

            # Skala automatyczna Y oraz stała skala X (typowa dla fal wodnych)
            self.ax.set_xlim(0.1, 5.0)
            self.ax.set_ylim(0, np.max(self.spectrum_y) * 1.1)

            self.ax.grid(True, linestyle='--', alpha=0.5)

        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        plot_img = np.asarray(buf)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        return plot_img


def main():
    fps_assumption = 30.0
    analyzer = WaveAnalyzer(width=640, height=480, fps=fps_assumption)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, old_frame = cap.read()
    if not ret:
        return

    old_gray = analyzer.preprocess(old_frame)
    p0 = analyzer.extract_initial_features(old_gray)
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = analyzer.preprocess(frame)
        analyzer.update_wave_parameters(frame_gray)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **analyzer.lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                total_distance = 0.0
                valid_points = len(good_new)

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    total_distance += np.sqrt((a - c) ** 2 + (b - d) ** 2)
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), analyzer.color[i % 1000].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, analyzer.color[i % 1000].tolist(), -1)

                avg_pixel_speed = (total_distance / valid_points) if valid_points > 0 else 0
                img = cv2.add(frame, mask)
                speed_px_s = avg_pixel_speed * fps_assumption

                # --- RAMKA Z WYNIKAMI ---
                cv2.rectangle(img, (10, 10), (450, 150), (0, 0, 0), -1)

                cv2.putText(img, f"Predkosc: {speed_px_s:.1f} px/s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(img, f"Szczytowa czestotliwosc: {analyzer.current_f:.2f} Hz", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"Okres fali: {analyzer.current_T:.2f} s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(img, f"Szacowana dlugosc: {analyzer.current_L:.2f} m", (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                # --- GENEROWANIE I SKLEJANIE Z WYKRESEM FFT ---
                plot_img = analyzer.get_spectrum_image()

                # Zmiana rozmiaru wykresu, by pasował wysokością do okna kamery
                plot_img = cv2.resize(plot_img, (400, 480))

                # Połączenie obrazu wideo i wykresu
                combined_view = np.hstack((img, plot_img))

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                cv2.imshow('ZR_B: System Analizy Fal', combined_view)
        else:
            p0 = analyzer.extract_initial_features(frame_gray)
            cv2.imshow('ZR_B: System Analizy Fal', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            p0 = analyzer.extract_initial_features(frame_gray)
            mask = np.zeros_like(old_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
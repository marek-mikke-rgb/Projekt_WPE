import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def sine_func(x, A, w, phi, C):
    return A * np.sin(w * x + phi) + C


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)

    # --- OKNO I SLIDERY ---
    cv2.namedWindow("Edges")
    cv2.createTrackbar("T1", "Edges", 50, 500, nothing)
    cv2.createTrackbar("T2", "Edges", 150, 500, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- PROGI Z SLIDERÓW ---
        t1 = cv2.getTrackbarPos("T1", "Edges")
        t2 = cv2.getTrackbarPos("T2", "Edges")

        edges = cv2.Canny(blur, t1, t2)

        # --- PROFIL PIONOWY ---
        h, w = edges.shape
        x_line = w // 2
        profile = edges[:, x_line].astype(float)

        # --- SZCZYTY ---
        peaks, _ = find_peaks(profile, height=50, distance=20)

        if len(peaks) > 1:
            distances = np.diff(peaks)
            avg_distance = float(np.mean(distances))
        else:
            avg_distance = 0.0

        # --- APROKSYMACJA SINUSEM ---
        y = np.arange(len(profile))
        fit_ok = True

        try:
            guess = [50, 0.05, 0, np.mean(profile)]
            params, _ = curve_fit(sine_func, y, profile, p0=guess, maxfev=5000)
            fitted = sine_func(y, *params)

            if np.max(np.abs(fitted)) < 1e-6:
                fit_ok = False

        except Exception:
            fit_ok = False

        # --- FFT (widmo amplitudowe) ---
        fft_vals = np.fft.rfft(profile)
        fft_amp = np.abs(fft_vals)
        fft_amp = fft_amp / (fft_amp.max() + 1e-6)

        freqs = np.fft.rfftfreq(len(profile), d=1.0)

        # --- WIZUALIZACJA FFT ---
        fft_plot_w = 400
        fft_plot_h = 200
        fft_plot = np.zeros((fft_plot_h, fft_plot_w, 3), dtype=np.uint8)

        x_scaled = (freqs / freqs.max() * (fft_plot_w - 1)).astype(int)
        y_scaled = (fft_amp * (fft_plot_h - 1)).astype(int)

        for i in range(len(x_scaled) - 1):
            cv2.line(
                fft_plot,
                (x_scaled[i], fft_plot_h - 1 - y_scaled[i]),
                (x_scaled[i + 1], fft_plot_h - 1 - y_scaled[i + 1]),
                (0, 255, 255),
                1
            )

        # --- siatka częstotliwości ---
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            xf = int(f * (fft_plot_w - 1))
            cv2.line(fft_plot, (xf, 0), (xf, fft_plot_h), (50, 50, 50), 1)
            cv2.putText(fft_plot, f"{f:.2f}", (xf + 2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- siatka amplitudy ---
        for a in [0.0, 0.5, 1.0]:
            ya = int((1 - a) * (fft_plot_h - 1))
            cv2.line(fft_plot, (0, ya), (fft_plot_w, ya), (50, 50, 50), 1)
            cv2.putText(fft_plot, f"{a:.1f}", (5, ya - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- WIZUALIZACJA PROFILU ---
        plot_w = 400
        plot_h = h
        plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)

        if profile.max() > 0:
            prof_scaled = (profile / profile.max()) * (plot_w - 1)
        else:
            prof_scaled = np.zeros_like(profile)

        if fit_ok and fitted.max() > 0:
            fit_scaled = (fitted / fitted.max()) * (plot_w - 1)
        else:
            fit_scaled = np.zeros_like(profile)

        prof_scaled = prof_scaled.astype(int)
        fit_scaled = fit_scaled.astype(int)

        for i in range(h - 1):
            cv2.line(plot,
                     (prof_scaled[i], i),
                     (prof_scaled[i + 1], i + 1),
                     (255, 255, 255), 1)

            if fit_ok:
                cv2.line(plot,
                         (fit_scaled[i], i),
                         (fit_scaled[i + 1], i + 1),
                         (0, 255, 0), 1)

        for p in peaks:
            cv2.circle(plot, (prof_scaled[p], p), 5, (0, 0, 255), -1)

        # --- RYSOWANIE SZCZYTÓW NA OBRAZIE ---
        vis = frame.copy()
        cv2.line(vis, (x_line, 0), (x_line, h), (0, 255, 0), 1)

        for p in peaks:
            cv2.circle(vis, (x_line, int(p)), 6, (0, 0, 255), -1)

        cv2.putText(vis, f"Szczyty: {len(peaks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, f"Srednia odleglosc: {avg_distance:.1f}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- WYŚWIETLANIE ---
        cv2.imshow("Frame", vis)
        cv2.imshow("Edges", edges)
        cv2.imshow("Vertical Profile + Sine Fit", plot)
        cv2.imshow("FFT Spectrum", fft_plot)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Optional
import numpy.fft as fft
import time

g = 9.81  # [m/s^2]


@dataclass
class CameraCalibration:
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    H: np.ndarray


@dataclass
class WaveParameters:
    wavelength: float
    period: float
    phase_speed: float
    direction_deg: float
    k: float
    omega: float


def get_example_calibration() -> CameraCalibration:
    K = np.eye(3, dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    return CameraCalibration(K=K, dist=dist, R=R, t=t, H=H)


def warp_to_world_plane(frame: np.ndarray, calib: CameraCalibration, out_size=(400, 200)) -> np.ndarray:
    return cv2.resize(frame, out_size)


# ============================================================
#   PIONOWY TIMESTACK (fale płyną z góry na dół)
# ============================================================

class TimeStackBuilder:
    def __init__(self, max_frames: int, line_x: int):
        self.max_frames = max_frames
        self.line_x = line_x
        self.stack = None

    def add_frame(self, world_gray: np.ndarray):
        col = world_gray[:, self.line_x].astype(np.float32)

        if self.stack is None:
            self.stack = col[None, :]
        else:
            self.stack = np.vstack([self.stack, col[None, :]])
            if self.stack.shape[0] > self.max_frames:
                self.stack = self.stack[-self.max_frames:, :]

    def get_stack(self):
        return self.stack


# ============================================================
#   FILTRACJA
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_temporal_filter(stack: np.ndarray, fps: float, lowcut=0.05, highcut=1.0):
    if stack is None:
        return None
    if stack.shape[0] < 30:
        return stack.copy()

    b, a = butter_bandpass(lowcut, highcut, fps, order=4)
    filtered = np.zeros_like(stack)
    for x in range(stack.shape[1]):
        col = stack[:, x]
        filtered[:, x] = filtfilt(b, a, col)
    return filtered


# ============================================================
#   WIDMO k–ω
# ============================================================

def compute_k_omega_spectrum(stack: np.ndarray, dx: float, dt: float):
    if stack is None:
        return None, None, None

    nt, nx = stack.shape
    win_t = np.hanning(nt)[:, None]
    win_x = np.hanning(nx)[None, :]
    windowed = stack * win_t * win_x

    F = fft.fft2(windowed)
    F_shift = fft.fftshift(F)
    S = np.abs(F_shift) ** 2

    freq_t = fft.fftfreq(nt, d=dt)
    freq_x = fft.fftfreq(nx, d=dx)

    freq_t_shift = fft.fftshift(freq_t)
    freq_x_shift = fft.fftshift(freq_x)

    omega = 2 * np.pi * freq_t_shift
    k = 2 * np.pi * freq_x_shift

    return S, k, omega


def find_dominant_wave_from_spectrum(S: np.ndarray, k: np.ndarray, omega: np.ndarray, depth: float) -> Optional[WaveParameters]:
    if S is None:
        return None

    idx = np.unravel_index(np.argmax(S), S.shape)
    it, ix = idx

    omega_axis = omega[it]
    k_axis = k[ix]

    if k_axis == 0:
        return None

    T = 2 * np.pi / abs(omega_axis)
    wavelength = 2 * np.pi / abs(k_axis)
    c = abs(omega_axis / k_axis)

    return WaveParameters(
        wavelength=wavelength,
        period=T,
        phase_speed=c,
        direction_deg=0.0,
        k=k_axis,
        omega=omega_axis
    )


# ============================================================
#   WYSOKOŚĆ FALI (szacowana)
# ============================================================

def estimate_wave_height_from_stack(stack: np.ndarray) -> float:
    if stack is None:
        return 0.0

    signal = np.mean(stack, axis=1)
    signal = signal - np.mean(signal)
    rms = np.sqrt(np.mean(signal ** 2))

    return rms * 0.05  # do kalibracji


# ============================================================
#   GŁÓWNY PROGRAM
# ============================================================

def main():
    video_path = "test_wave_vertical.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0
    dt = 1.0 / fps

    calib = get_example_calibration()

    out_w = 400
    out_h = 200

    # --- KLUCZOWA ZMIANA ---
    # 1 px = 1 m (w sensie analizy)
    dx = 1.0

    line_x = int(out_w * 0.5)
    max_frames_stack = 512
    stack_builder = TimeStackBuilder(max_frames=max_frames_stack, line_x=line_x)

    last_estimate_time = 0
    estimate_interval = 2.0

    wave_params = None
    height_est = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        world = warp_to_world_plane(frame_gray, calib, out_size=(out_w, out_h))

        stack_builder.add_frame(world)
        stack = stack_builder.get_stack()

        now = time.time()

        if stack is not None and (now - last_estimate_time) > estimate_interval:
            last_estimate_time = now

            filtered_stack = apply_temporal_filter(stack, fps=fps)
            S, k_axis, omega_axis = compute_k_omega_spectrum(filtered_stack, dx=dx, dt=dt)
            wave_params = find_dominant_wave_from_spectrum(S, k_axis, omega_axis, depth=1.0)
            height_est = estimate_wave_height_from_stack(filtered_stack)

        # --- OKNO PARAMETRÓW ---
        param_window = np.zeros((260, 350, 3), dtype=np.uint8)

        def put(txt, y):
            cv2.putText(param_window, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        if wave_params is not None:
            put(f"lambda: {wave_params.wavelength:.3f} px", 40)
            put(f"T: {wave_params.period:.3f} s", 80)
            put(f"c: {wave_params.phase_speed:.3f} px/s", 120)
            put(f"k: {wave_params.k:.3f} 1/px", 160)
            put(f"omega: {wave_params.omega:.3f} rad/s", 200)

        if height_est > 0:
            put(f"H~: {height_est:.3f} px", 240)

        cv2.imshow("Wave Parameters", param_window)

        # --- OBRAZ ---
        vis = cv2.cvtColor(world, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (line_x, 0), (line_x, out_h - 1), (0, 255, 0), 1)
        cv2.imshow("World (XY) + line", vis)

        # --- TIMESTACK ---
        if stack is not None:
            st_vis = stack.copy()
            st_vis = st_vis - st_vis.min()
            if st_vis.max() > 0:
                st_vis = st_vis / st_vis.max()
            st_vis = (st_vis * 255).astype(np.uint8)
            st_vis_color = cv2.applyColorMap(st_vis, cv2.COLORMAP_JET)
            cv2.imshow("Timestack (t vs y)", st_vis_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

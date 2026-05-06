import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Optional
import numpy.fft as fft
import math
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
    # Na razie: brak realnej kalibracji → H = I
    K = np.eye(3, dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    return CameraCalibration(K=K, dist=dist, R=R, t=t, H=H)


def warp_to_world_plane(frame: np.ndarray, calib: CameraCalibration, out_size=(400, 200)) -> np.ndarray:
    # Zamiast prawdziwego rzutu: po prostu skalujemy obraz
    frame_resized = cv2.resize(frame, out_size)
    return frame_resized


class TimeStackBuilder:
    def __init__(self, max_frames: int, line_y: int):
        self.max_frames = max_frames
        self.line_y = line_y
        self.stack = None

    def add_frame(self, world_gray: np.ndarray):
        row = world_gray[self.line_y, :].astype(np.float32)
        if self.stack is None:
            self.stack = row[None, :]
        else:
            self.stack = np.vstack([self.stack, row[None, :]])
            if self.stack.shape[0] > self.max_frames:
                self.stack = self.stack[-self.max_frames:, :]

    def get_stack(self):
        return self.stack


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

    direction_deg = 0.0  # 1D timestack

    return WaveParameters(
        wavelength=wavelength,
        period=T,
        phase_speed=c,
        direction_deg=direction_deg,
        k=k_axis,
        omega=omega_axis
    )


def main():
    video_path = "waves_complex.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0
    dt = 1.0 / fps

    calib = get_example_calibration()

    world_width_m = 10.0
    out_w = 400
    out_h = 200
    dx = world_width_m / out_w
    depth_h = 1.5

    # linia w dolnej części obrazu (np. 70% wysokości) – tam będą fale
    line_y = int(out_h * 0.7)
    max_frames_stack = 512
    stack_builder = TimeStackBuilder(max_frames=max_frames_stack, line_y=line_y)

    last_estimate_time = 0
    estimate_interval = 2.0

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
        wave_params = None

        if stack is not None and (now - last_estimate_time) > estimate_interval:
            last_estimate_time = now

            filtered_stack = apply_temporal_filter(stack, fps=fps)
            S, k_axis, omega_axis = compute_k_omega_spectrum(filtered_stack, dx=dx, dt=dt)
            wave_params = find_dominant_wave_from_spectrum(S, k_axis, omega_axis, depth=depth_h)

        vis = cv2.cvtColor(world, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (0, line_y), (out_w - 1, line_y), (0, 255, 0), 1)

        if wave_params is not None:
            cv2.putText(vis, f"lambda: {wave_params.wavelength:.3f} m", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis, f"T: {wave_params.period:.3f} s", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis, f"c: {wave_params.phase_speed:.3f} m/s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("World (XY) + line", vis)

        if stack is not None:
            st_vis = stack.copy()
            st_vis = st_vis - st_vis.min()
            if st_vis.max() > 0:
                st_vis = st_vis / st_vis.max()
            st_vis = (st_vis * 255).astype(np.uint8)
            st_vis_color = cv2.applyColorMap(st_vis, cv2.COLORMAP_JET)
            cv2.imshow("Timestack (t vs x)", st_vis_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

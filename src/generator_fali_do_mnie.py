import cv2
import numpy as np

# Parametry obrazu
width = 900
height = 600
fps = 30
duration = 12
frames = fps * duration

# Parametry fali (zgodne z analizatorem)
lambda_px = 80          # długość fali w pikselach
T = 1.0                 # okres fali w sekundach
c = lambda_px / T       # prędkość fazowa px/s
dy_per_frame = c / fps  # przesunięcie fali na klatkę
amplitude = 20          # wysokość fali w px

# Siatka pionowa (fala płynie z góry na dół)
y = np.linspace(0, height, height)[:, None]
x = np.linspace(0, width, width)[None, :]

# Częstość kołowa i liczba falowa
k = 2 * np.pi / lambda_px
omega = 2 * np.pi / T

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_wave_vertical.mp4", fourcc, fps, (width, height))

for t in range(frames):

    # Faza fali (propagacja w dół)
    phase = k * y - omega * (t / fps)

    # Czysta fala sinusoidalna
    H = amplitude * np.sin(phase)

    # Normalizacja do 0–255
    H_norm = (H - H.min()) / (H.max() - H.min())
    H_img = (H_norm * 255).astype(np.uint8)

    # Kolor wody (niebieski)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = H_img  # blue
    frame[:, :, 1] = H_img // 2
    frame[:, :, 2] = H_img // 4

    out.write(frame)

out.release()
print("Gotowe! Plik test_wave_vertical.mp4 został wygenerowany.")

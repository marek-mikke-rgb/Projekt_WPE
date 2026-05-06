import cv2
import numpy as np

# Parametry filmu
width = 800
height = 300
fps = 30
duration = 15  # sekundy
frames = fps * duration

# Parametry fal (superpozycja)
waves = [
    {"amp": 35, "wl": 180, "speed": 1.2},   # duża, wolna fala
    {"amp": 15, "wl": 90,  "speed": 2.0},   # średnia fala
    {"amp": 8,  "wl": 45,  "speed": 3.5},   # mała, szybka fala
]

# Szum powierzchni
noise_strength = 4

# Tworzenie pliku wideo
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("waves_complex.mp4", fourcc, fps, (width, height))

for t in range(frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    x = np.arange(width)
    y = np.zeros_like(x, dtype=float)

    # Superpozycja fal
    for wv in waves:
        y += wv["amp"] * np.sin(2 * np.pi * (x - wv["speed"] * t) / wv["wl"])

    # Dodanie szumu
    y += np.random.normal(0, noise_strength, size=width)

    # Przesunięcie na środek obrazu
    y += height // 2

    # Rysowanie fali
    for i in range(width - 1):
        cv2.line(frame,
                 (i, int(y[i])),
                 (i + 1, int(y[i + 1])),
                 (255, 255, 255), 2)

    out.write(frame)

out.release()
print("Gotowe! Plik waves_complex.mp4 został wygenerowany.")

import cv2
import numpy as np

# Parametry obrazu
width = 900
height = 600
fps = 30
duration = 12
frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("single_strong_wave.mp4", fourcc, fps, (width, height))

# Siatka współrzędnych
x = np.linspace(-1, 1, width)[None, :]
y = np.linspace(0, 1, height)[:, None]
X, Y = np.meshgrid(x[0, :], y[:, 0], indexing="xy")

# Perspektywa kamery
persp = 0.6 + 2.0 * (1 - Y)       # większa amplituda w dole
scale = 0.5 + 1.2 * (1 - Y)       # fale gęstsze w oddali

# JEDNA FALA
amp = 1.8        # mocna fala
wl = 0.16        # długość fali
speed = 1.4      # prędkość
angle = 0.15     # lekkie nachylenie grzbietów

# Światło
light_dir = np.array([0.25, -0.6, 1.0])
light_dir = light_dir / np.linalg.norm(light_dir)

for t in range(frames):

    # --- MAPA WYSOKOŚCI (jedna fala) ---
    kx = np.sin(angle)
    ky = np.cos(angle)

    phase = 2 * np.pi * (
        (X * kx + Y * ky * scale) / wl
        - t * speed / fps
    )

    H = amp * np.sin(phase)

    # Szum powierzchni (delikatny)
    H += 0.05 * np.random.normal(0, 1, H.shape)

    # Perspektywa
    H *= persp

    # --- NORMALNE POWIERZCHNI ---
    dx = np.gradient(H, axis=1)
    dy = np.gradient(H, axis=0)

    normals = np.dstack((-dx, -dy, np.ones_like(H)))
    normals /= (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6)

    # --- OŚWIETLENIE ---
    diffuse = np.clip(np.sum(normals * light_dir, axis=2), 0, 1)
    spec = np.power(np.clip(diffuse, 0, 1), 30)

    # Kolor wody – jasny, turkusowy
    base = 90 + 120 * diffuse
    blue = base + 150 * spec
    green = base * 0.9
    red = base * 0.55

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.clip(blue, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(green, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(red, 0, 255).astype(np.uint8)

    # lekkie rozmycie
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    out.write(frame)

out.release()
print("Gotowe! Plik single_strong_wave.mp4 został wygenerowany.")

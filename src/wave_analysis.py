import cv2
import numpy as np
from scipy.signal import find_peaks


def find_best_wave_line(edges, step=10):
    h, w = edges.shape
    best_y = 0
    best_score = -1

    for y in range(0, h, step):
        score = np.sum(edges[y, :])
        if score > best_score:
            best_score = score
            best_y = y

    return best_y


def analyze_wave_profile(frame, prev_edges=None, alpha=0.6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ROI – dolna część obrazu (np. dolne 60%)
    h, w = gray.shape
    y0 = int(h * 0.4)
    gray_roi = gray[y0:h, :]

    # lekkie wygładzenie – tylko Gaussian, bez CLAHE, bez median
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Canny – zacznij od takich progów, potem możesz je stroić
    edges = cv2.Canny(gray_roi, 100, 200)

    # opcjonalna stabilizacja krawędzi
    if prev_edges is not None:
        edges = cv2.addWeighted(prev_edges, alpha, edges, 1 - alpha, 0)

    # dynamiczna linia
    y_local = find_best_wave_line(edges, step=10)
    y_global = y_local + y0

    # profil
    profile = edges[y_local, :].astype(np.float32)

    # brak agresywnego wygładzania – używamy profilu „as is”
    profile_smooth = profile.copy()

    # bezpieczne wykrywanie szczytów
    if len(profile_smooth) < 10 or np.max(profile_smooth) < 10:
        peaks = np.array([])
    else:
        try:
            peaks, _ = find_peaks(
                profile_smooth,
                distance=20,
                prominence=10,
                height=10
            )
        except Exception:
            peaks = np.array([])

    # edges w pełnym obrazie
    edges_full = np.zeros_like(gray)
    edges_full[y0:y0 + edges.shape[0], :] = edges

    return {
        "edges": edges_full,
        "profile": profile,
        "profile_smooth": profile_smooth,
        "peaks": peaks,
        "y": y_global,
        "edges_raw": edges
    }

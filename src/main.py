import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from wave_analysis import analyze_wave_profile


def main():
    cap = cv2.VideoCapture(0)

    plt.ion()
    fig, ax = plt.subplots()
    line_raw, = ax.plot([], [], label="profil", color="red")
    ax.set_ylim(0, 260)
    ax.set_xlim(0, 640)
    ax.set_title("Profil fali")
    ax.set_xlabel("pozycja [px]")
    ax.set_ylabel("intensywnosc krawedzi")
    ax.legend()

    prev_edges = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = analyze_wave_profile(frame, prev_edges=prev_edges)
        prev_edges = result["edges_raw"]

        edges = result["edges"]
        profile = result["profile"]
        peaks = result["peaks"]
        y = result["y"]

        cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 1)

        for p in peaks:
            cv2.circle(frame, (int(p), y), 5, (0, 0, 255), -1)

        x_vals = np.arange(len(profile))
        line_raw.set_xdata(x_vals)
        line_raw.set_ydata(profile)

        ax.set_xlim(0, len(profile) if len(profile) > 0 else 640)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        cv2.imshow("Fale - obraz", frame)
        cv2.imshow("Fale - krawedzie", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()

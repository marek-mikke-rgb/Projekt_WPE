# System wizyjny do analizy parametrów fal morskich

System do analizy falowania wody z wideo. Zbudowany z myślą o płynnym działaniu na Raspberry Pi 5. Program wykorzystuje OpenCV i SciPy do wyciągania parametrów fali w czasie rzeczywistym, a do tego ma wbudowany HUD i prostą symulację łodzi (inercja po zadaniu prędkości).

## Funkcje
* **FFT:** Program analizuje jasność środka kadru i wylicza szczytową częstotliwość fali, okres i szacowaną długość.
* **Optical Flow:** Śledzenie punktów (Lucas-Kanade + SIFT) pozwala na pomiar prędkości nurtu wody w pikselach na sekundę.
* **UI:** Cały HUD (wykresy, paski, manetka) jest rysowany natywnie na macierzach NumPy.
* **Symulacja łodzi:** Można zmieniać nastawę manetki na ekranie. Skrypt oblicza inercję pierwszego rzędu, więc łódź przyspiesza płynnie.

## Uruchomienie na Raspberry Pi 5
```bash
Instalacja:
   python3 -m venv venv
   source venv/bin/activate
   pip install opencv-python numpy scipy
   
Uruchomienie:
   python main.py

Możliwa personalizacja w main.py:
   SOURCE = "ścieżka_do_filmu.mp4" – analiza nagrania.
   SOURCE = 0 – analiza obrazu z kamery na żywo.

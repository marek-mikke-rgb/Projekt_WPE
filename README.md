# System wizyjny do analizy parametrów fal morskich

System do analizy falowania wody z wideo. Zbudowany z myślą o płynnym działaniu na Raspberry Pi 5. Program wykorzystuje OpenCV i SciPy do wyciągania parametrów fali w czasie rzeczywistym, a do tego ma wbudowany HUD i prostą symulację łodzi (inercja po zadaniu prędkości).

## Funkcje
* **FFT:** Program analizuje jasność środka kadru i wylicza szczytową częstotliwość fali, okres i szacowaną długość.
* **Optical Flow:** Śledzenie punktów (Lucas-Kanade + SIFT) pozwala na pomiar prędkości nurtu wody w pikselach na sekundę.
* **UI:** Cały HUD (wykresy, paski, manetka) jest rysowany natywnie na macierzach NumPy.
* **Symulacja łodzi:** Można zmieniać nastawę manetki na ekranie. Skrypt oblicza inercję pierwszego rzędu, więc łódź przyspiesza płynnie.

## Uruchomienie na Raspberry Pi 5
```bash
WPE – Instrukcja uruchamiania programów na Raspberry Pi

Łączenie przez SSH – w terminalu (wiersz poleceń): 
   Należy połączyć się z Internetem wykorzystując kabel Ethernet.
   W terminalu należy wpisać: 
   ssh użytkownik@adres_ip
   Pojawi się konieczność wpisania hasła: twoje_hasło

Tworzenie środowiska wirtualnego i pierwsze połączenie (raspberry): 
   cd ścieżka_do_projektu_na_Pi5
   Pomocna może być komenda ls do sprawdzenia zawartości folderu. Upewnij się, że jest plik 
   requirements.txt
   python3 –m venv –system-site-packages .venv
   Powyższa komenda pozwala na korzystanie z bibliotek systemowych
   source .venv/bin/activate
   Po tej komendzie po lewej stronie znaku zachęty powinien pojawić się napis (.venv)
   pip install --upgrade pip
   pip install –r requirements.txt
   cd src
   python nazwa_pliku.py

Kolejne połączenia (raspberry):
   cd ścieżka_do_projektu_na_Pi5
   source .venv/bin/activate
   cd src
   python nazwa_pliku.py

Zakończenie połączenia (raspberry): 
   Aby zatrzymać obecnie działający skrypt należy wcisnąć kombinację Ctrl + C
   cd ..
   deactivate
   exit

Usuwanie środowiska wirtualnego (raspberry): 
   Uwaga, środowisku musi być nieaktywne. W przypadku, jeśli wcześniej uruchamiano program 
   pythona, należy wpierw zakończyć połączenie!
   cd ścieżka_do_projektu_na_Pi5
   rm -rf .venv

Tworzenie requirements.txt (terminal pycharm): 
   W terminalu w np. pycharmie należy wprowadzić:
   pip freeze > requirements.txt

Wysyłanie plików przez SSH (wiersz poleceń): 
   scp –r ścieżka_projektu_na_komputerze
   użytkownik@adres_ip:ścieżka_do_projektu_na_Pi5

Możliwa personalizacja w main.py:
   SOURCE = "ścieżka_do_filmu.mp4" – analiza nagrania.
   SOURCE = 0 – analiza obrazu z kamery na żywo.

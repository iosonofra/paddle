# PaddleOCR Studio 🚀

Un'applicazione web locale moderna, veloce e completa per l'estrazione e il riconoscimento di testo da immagini e documenti PDF, basata su **PaddleOCR** e **FastAPI**.

---

## ✨ Funzionalità

- **Riconoscimento OCR Completo**: Basato sugli ultimi modelli PaddleOCR (PP-OCRv4) per rilevamento e decodifica dei caratteri.
- **Supporto Documenti & Immagini**:
  - Immagini: JPG, PNG, WEBP, TIFF, BMP.
  - Documenti: PDF multipagina con navigatore per pagina.
- **Visualizzatore Interattivo (Canvas)**:
  - Bounding box colorati per livello di confidenza (Verde > 85%, Giallo 60-85%, Rosso < 60%).
  - Zoom e Pan con rotellina del mouse e trascinamento.
  - Evidenziazione bidirezionale: posiziona il cursore su un box per evidenziare la riga corrispondente e viceversa.
- **Multilingua**: Supporto integrato per oltre 10 lingue (Italiano, Inglese, Francese, Tedesco, Spagnolo, Cinese, Giapponese, Coreano, ecc.).
- **Rotazione Automatica (`angle_cls`)**: Riconosce e corregge testi inclinati o capovolti.
- **Ricerca Rapida**: Filtra e trova istantaneamente parole o cifre nei riquadri rilevati.
- **Incolla da Appunti**: Premi <kbd>Ctrl</kbd> + <kbd>V</kbd> ovunque per analizzare uno screenshot catturato al volo.
- **Esportazione Multipla**:
  - Copia negli appunti con un clic.
  - Download in formato `.txt` (testo continuo).
  - Download in formato `.json` (dati grezzi e coordinate).
  - Download in formato `.csv` (tabella con pagina, testo, confidenza e box).
- **Esempi Integrati**: 3 sample pronti all'uso (Scontrino, Fattura, Cartello) per testare l'app in 1 secondo.

---

## 🚀 Come Eseguire l'App

### Modalità 1-Click (Windows)
Fai doppio clic sul file:
```text
run.bat
```
Questo script:
1. Crea/verifica l'ambiente virtuale isolato (`.venv`) basato su Python 3.10.
2. Verifica e installa le dipendenze da `requirements.txt`.
3. Avvia il server FastAPI e apre automaticamente il browser all'indirizzo `http://127.0.0.1:8000`.

### Modalità PowerShell
```powershell
.\run.ps1
```

### Modalità Manuale
```powershell
# Attiva il virtualenv
.\.venv\Scripts\activate

# Avvia il server
python server.py
```
Poi apri il browser su: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🛠️ API REST

L'applicazione espone anche endpoint API REST:
- `POST /api/ocr`: accetta `file` (multipart), `lang` (default `it`), `use_angle_cls` (bool), `min_confidence` (float). Restituisce coordinate, confidenze e testo estratto.
- `GET /api/languages`: lista delle lingue supportate.
- `GET /api/system`: informazioni sullo stato dell'hardware (GPU/CPU).
- Documentazione Swagger interattiva disponibile su: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

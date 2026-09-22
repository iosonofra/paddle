#!/usr/bin/env bash
# ==============================================================================
# PaddleOCR Studio - Script di Installazione Completo per Debian / Proxmox LXC
# ==============================================================================

set -e

# Controllo permessi root o sudo
if [ "$EUID" -ne 0 ]; then
    echo "Questo script richiede i privilegi di amministratore. Esegui con: sudo bash install.sh"
    exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-root}"

echo ""
echo "=============================================================================="
echo "          Installazione PaddleOCR Studio su Debian / Proxmox                  "
echo "=============================================================================="
echo "Cartella applicazione: $APP_DIR"
echo "Utente del servizio:   $TARGET_USER"
echo ""

# 1. Installazione librerie di sistema Debian
echo ">>> [1/5] Installazione dipendenze di sistema C/C++ (glibc, OpenGL, GOMP)..."
apt-get update -y
apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 libgomp1 curl git

# 2. Rilevamento e selezione della migliore versione di Python (preferite 3.10 / 3.11 / 3.12)
PYTHON_BIN="python3"
SYS_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "11")

if [ "$SYS_MINOR" -ge 13 ]; then
    echo ">>> Rilevato Python 3.$SYS_MINOR (molto recente). Verifica disponibilità Python 3.11 / 3.12..."
    apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null || true
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
        echo ">>> Utilizzo di python3.11 per massima stabilità con i pacchetti AI."
    elif command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        echo ">>> Utilizzo di python3.12."
    fi
fi

# 3. Creazione ambiente virtuale isolato
echo ">>> [2/5] Configurazione ambiente virtuale con $PYTHON_BIN in .venv..."
rm -rf "$APP_DIR/.venv"
$PYTHON_BIN -m venv "$APP_DIR/.venv"

# 4. Installazione dipendenze Python
echo ">>> [3/5] Installazione pacchetti Python (PaddlePaddle, FastAPI, ReportLab)..."
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# 5. Pre-inizializzazione modelli per il primo avvio rapido
echo ">>> [4/5] Pre-scaricamento modelli OCR (per avvio immediato)..."
"$APP_DIR/.venv/bin/python" -c "
try:
    from paddleocr import PaddleOCR
    import numpy as np
    ocr = PaddleOCR(use_angle_cls=True, lang='it', use_gpu=False)
    dummy = np.ones((50, 50, 3), dtype=np.uint8) * 255
    ocr.ocr(dummy, cls=True)
    print('Modelli pre-caricati con successo!')
except Exception as e:
    print('Avviso pre-caricamento modelli:', e)
" || true

# 6. Registrazione e avvio del servizio di sistema Systemd
echo ">>> [5/5] Configurazione servizio systemd (avvio automatico 24/7)..."
SERVICE_FILE="/etc/systemd/system/paddleocr.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=PaddleOCR Studio Web Application
After=network.target

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$APP_DIR
Environment="OMP_NUM_THREADS=4"
Environment="MKL_NUM_THREADS=4"
Environment="FLAGS_use_mkldnn=1"
Environment="FLAGS_allocator_strategy=auto_growth"
ExecStart=$APP_DIR/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable paddleocr.service
systemctl restart paddleocr.service

# Rilevamento indirizzo IP locale
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=============================================================================="
echo " ✅ INSTALLAZIONE COMPLETATA CON SUCCESSO!"
echo "=============================================================================="
echo " L'applicazione è ora attiva e in esecuzione come servizio systemd."
echo ""
echo " 🌐 Accedi alla Web App dal browser:"
echo "    http://${LOCAL_IP:-localhost}:8000"
echo ""
echo " 📋 Comandi utili:"
echo "    Stato servizio:   sudo systemctl status paddleocr"
echo "    Riavvia servizio: sudo systemctl restart paddleocr"
echo "    Log in diretta:   sudo journalctl -u paddleocr -f"
echo "    Aggiorna con git: sudo bash update.sh"
echo "=============================================================================="

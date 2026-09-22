#!/usr/bin/env bash
# ==============================================================================
# Script di Installazione e Configurazione Servizio PaddleOCR Studio su Debian/Ubuntu
# ==============================================================================

set -e

echo "=== [1/4] Aggiornamento pacchetti e installazione dipendenze C/C++ ==="
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 libgomp1 curl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== [2/4] Creazione ambiente virtuale Python ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "=== [3/4] Installazione dipendenze Python da requirements.txt ==="
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "=== [4/4] Creazione servizio systemd per esecuzione automatica continua ==="
SERVICE_PATH="/etc/systemd/system/paddleocr.service"
CURRENT_USER=$(whoami)

sudo bash -c "cat <<EOF > $SERVICE_PATH
[Unit]
Description=PaddleOCR Studio Web Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
Environment="OMP_NUM_THREADS=4"
Environment="MKL_NUM_THREADS=4"
Environment="FLAGS_use_mkldnn=1"
Environment="FLAGS_allocator_strategy=auto_growth"
ExecStart=$SCRIPT_DIR/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable paddleocr
sudo systemctl restart paddleocr

echo ""
echo "=============================================================================="
echo " PaddleOCR Studio installato con successo!"
echo " Servizio systemd attivo: sudo systemctl status paddleocr"
echo " L'applicazione è disponibile all'indirizzo: http://$(hostname -I | awk '{print $1}'):8000"
echo "=============================================================================="

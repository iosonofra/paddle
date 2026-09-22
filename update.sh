#!/usr/bin/env bash
# ==============================================================================
# PaddleOCR Studio - Script di Aggiornamento Rapido (Git Pull + Restart)
# ==============================================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Questo script richiede i privilegi di amministratore. Esegui con: sudo bash update.sh"
    exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

echo ">>> [1/3] Download aggiornamenti da repository Git..."
git pull

echo ">>> [2/3] Aggiornamento dipendenze Python..."
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo ">>> [3/3] Riavvio servizio PaddleOCR Studio..."
systemctl restart paddleocr.service

echo ""
echo "✅ Aggiornamento completato con successo e servizio riavviato!"
systemctl status paddleocr.service --no-pager

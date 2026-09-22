#!/usr/bin/env bash
# ==============================================================================
# PaddleOCR Studio - Script di Disinstallazione Servizio
# ==============================================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Questo script richiede i privilegi di amministratore. Esegui con: sudo bash uninstall.sh"
    exit 1
fi

echo ">>> Arresto e disabilitazione servizio systemd..."
systemctl stop paddleocr.service || true
systemctl disable paddleocr.service || true
rm -f /etc/systemd/system/paddleocr.service
systemctl daemon-reload

echo ">>> Rimozione ambiente virtuale .venv..."
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "$APP_DIR/.venv"

echo "✅ Servizio e virtualenv rimossi con successo."

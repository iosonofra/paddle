FROM python:3.10-slim-bookworm

# Imposta variabili di ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Installa le librerie C/C++ di sistema essenziali per OpenCV e PaddlePaddle (glibc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia i file dell'applicazione
COPY . .

# Esponi la porta HTTP
EXPOSE 8000

# Controllo dello stato del servizio
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/system || exit 1

# Avvia FastAPI su 0.0.0.0 per renderlo accessibile dall'esterno
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

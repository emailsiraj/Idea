FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Install Tesseract + OpenCV runtime dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "0_Aadhar_Verification.py","--server.port=7860","--server.address=0.0.0.0","--server.headless=true"]

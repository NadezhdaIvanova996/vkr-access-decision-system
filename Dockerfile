FROM python:3.11-slim

# Устанавливаем все необходимые системные библиотеки для OpenCV + face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем файлы проекта
COPY app.py .
COPY extract_emotions_fer.py .
COPY train_risk_model.py .
COPY requirements.txt .
COPY models/ ./models/
COPY known_faces/ ./known_faces/

RUN mkdir -p data experiments figures

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
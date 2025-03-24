FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY app.py .
COPY templates/ ./templates/

COPY best_weights.pt .

RUN pip install --no-cache-dir -r requirements.txt

# patch
RUN sed -i "s/return torch.load(file, map_location='cpu')/return torch.load(file, map_location='cpu', weights_only=False)/" /usr/local/lib/python3.9/site-packages/ultralytics/nn/tasks.py

EXPOSE 8000
CMD ["python3", "app.py"]

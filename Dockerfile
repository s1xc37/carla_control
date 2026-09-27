# Наши скрипты: сцена (run_scene.py + pedestrians.py) и отправщик (ws_sender.py).
# Сервер CARLA 0.9.16 в образ не входит — он запускается на хосте (./CarlaUE4.sh).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
ENV PYTHONUNBUFFERED=1

CMD ["python", "run_scene.py"]

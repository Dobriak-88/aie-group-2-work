FROM python:3.10-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/

# Создаём папки для артефактов и логов
RUN mkdir -p /app/artifacts /app/logs

# Переменные окружения по умолчанию
ENV DATA_PATH=/app/src/data/cars.csv
ENV ARTIFACTS_PATH=/app/artifacts
ENV LOGS_PATH=/app/logs
ENV RANDOM_STATE=42

# Точка входа: python -m
ENTRYPOINT ["python", "-m"]

# Команда по умолчанию – показать справку CLI
CMD ["src.service.cli", "--help"]
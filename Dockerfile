FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: for librosa and audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Expose FastAPI default port
EXPOSE 8000

CMD ["bash", "start.sh"]

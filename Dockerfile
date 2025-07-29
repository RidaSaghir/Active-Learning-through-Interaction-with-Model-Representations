FROM python:3.11-slim

WORKDIR /app

#That makes /app (the Docker working dir) visible as a top-level package for imports like import config
ENV PYTHONPATH=/app

# copy everything to the working directory
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: for librosa and audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Expose FastAPI default port
EXPOSE 8000

CMD ["bash", "start.sh"]

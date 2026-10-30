FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN sed '/# Documentation/q' /tmp/requirements.txt > /tmp/core.txt && pip install --no-cache-dir -r /tmp/core.txt
ENV PYTHONPATH=/dicee
WORKDIR /dicee

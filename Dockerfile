FROM python:3.10
WORKDIR /dicee
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD ./main.py
LABEL org.opencontainers.image.source=https://github.com/dice-group/dice-embeddings

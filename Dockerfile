FROM --platform=linux/x86_64 python:3.8.12-buster

WORKDIR /prod
COPY catface catface.py

COPY packages.txt packages.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD uvicorn catface.api.api:app --host 0.0.0.0 --port $PORT

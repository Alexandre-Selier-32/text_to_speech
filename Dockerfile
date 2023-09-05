FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt
COPY bark_model /bark_model
COPY app/params.py /app/params.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn bark_model.bark_api:app --host 0.0.0.0

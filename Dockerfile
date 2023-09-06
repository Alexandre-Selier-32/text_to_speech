FROM python:3.10.6-buster

COPY requirements_prod.txt /requirements.txt
COPY Tacotron2_model /Tacotron2_model
COPY app/params.py /app/params.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn Tacotron2_model.Tacotron2_api:app --host 0.0.0.0 --port $PORT

EXPOSE 8080

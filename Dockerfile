FROM python:3.8-slim

WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yml ./config.yml
COPY training.yml ./training.yml

RUN rasa telemetry disable
RUN rasa train nlu --nlu ./training.yml

EXPOSE 5005
CMD rasa run --enable-api -m models

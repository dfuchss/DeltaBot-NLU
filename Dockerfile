FROM python:3.10.0

WORKDIR /usr/src/multi_nlu
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN rasa telemetry disable
COPY json_entity_extractor.py .

# DE Model
COPY de de
RUN rasa train nlu --config de/config.yml --nlu de/training.yml --out models_de

# EN Model
COPY en en
RUN rasa train nlu --config en/config.yml --nlu en/training.yml --out models_en

COPY multi_nlu.py .

EXPOSE 5005
ENTRYPOINT python multi_nlu.py

FROM rasa/rasa:2.8.2

COPY config.yml ./config.yml
COPY training.yml ./training.yml

# For Entities
COPY entities.json .
COPY json_entity_extractor.py .

RUN rasa telemetry disable
RUN rasa train nlu --nlu ./training.yml

EXPOSE 5005
ENTRYPOINT rasa run --enable-api -m models

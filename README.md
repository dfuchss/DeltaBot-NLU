⚠️ Archived because not needed for current version of DeltaBot ⚠️

# DeltaBot NLU (Version 2)

The NLU Unit for [DeltaBot V2](https://github.com/dfuchss/DeltaBot)

## Requirements (Development):

- python3 with pip
- `pip install -r requirements.txt`
- `rasa train nlu --config en/config.yml --nlu en/training.yml --out models_en`
- `rasa train nlu --config de/config.yml --nlu de/training.yml --out models_de`
- `python multi_nlu.py`

Tre main aim of this project was training a series of models in jax+haiku, to form a model zoo. The project also contains reusable logger, losses, and updater.

## Run
To run single model training modify the default configuration in config.py and run train.py.
To run model zoo training run zoo.py with desired arguments. The parameters that are not specified are sampled randomly as implemented in config.py.
To load a pretrained model zoo from a path use zoo_dataloader.load_nets(path)

## Installation
Clone this repo, then first install jax.
Then
```
pip install -e .
```


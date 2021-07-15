# Timbre Controlled Auto Encoder

Introduction

## Model

## Data

## Timbre Heuristics

## Training

The class `LocalConfig` must be used to describe training configurations. Eg.

```python
from tcvae.localconfig import LocalConfig

c = LocalConfig()
c.batch_size = 12
c.dataset_dir = "/path/to/dataset"
c.use_encoder = False
```

The configuration can be saved and loaded:

```python
c.save()
c.load_from_file(file_path)
```

To train a model:

```python
from tcvae import train, localconfig

c = localconfig.LocalConfig()
train.train(c)
```

## Prediction

## Deployment

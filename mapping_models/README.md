# Mapping Models

Models for mapping Text to Sound Outputs to DDSP Decoder Inputs

For a rapid experiment, just use the included trainer script. For example:

```python
import tensorflow as tf
from mapping_models import trainer

model = tf.keras.models.load_model('path/to/saved/model')

trainer.train(
    model,
    dataset_dir='your_dataset_dir',
    model_dir='model_dir_name'
)

```
Complete example: [GRU Test](examples/gru_test_model.py)

## Installation

Create a virtual environment (recommended)

`virtualenv venv`

Activate it

`venv\scripts\activate`

Install module in edit mode

`python -m pip install -e .`

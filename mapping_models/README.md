# Mapping Models

Models for mapping Text to Sound Outputs to DDSP Decoder Inputs

For a rapid experiment, just use the included trainer script. For example:

```python
import tensorflow as tf
from src import trainer


model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dense(2, activation='tanh')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.losses.MeanSquaredError()]
)


if __name__ == '__main__':
    trainer.train(
        model,
        dataset_dir='your_dataset_dir',
        model_dir='model_dir_name'
    )

```
Complete example: [GRU Test](src/gru_test_model.py)

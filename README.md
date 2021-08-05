# Timbre Conditioned (Variational) Auto Encoder

## Training

TBD

## Dataset

TBD

## Sound Generator

Get audio prediction

```python
from sound_generator import SoundGenerator


sg = SoundGenerator()

success, audio = sg.get_prediction({
    "input_pitch": 40,
    "pitch": 40,
    "velocity": 100,
    # A list of sg.conf.num_measures values between 0 and 1
    "heuristic_measures": [0.1] * sg.conf.num_measures,
    # A list of sg.conf.latent_dim values between 0 and 1
    "latent_sample": [0.5] * sg.conf.latent_dim
})

```

Required keys in the input dictionary:

**input_pitch**: Note number to use in decoder input

**pitch**: Note number to use in audio synthesis

**velocity**: Velocity of the note between 25 and 127

**latent_sample**: Values for z input to decoder

**heuristic_measures**: List of values for following measures used in decoder in the sequence shown:
```python
['inharmonicity',
 'even_odd',
 'sparse_rich',
 'attack_rms',
 'decay_rms',
 'attack_time',
 'decay_time',
 'bass',
 'mid',
 'high_mid',
 'high']
```

## Sound Generator App

Sound generator can also be used via a web app - [follow these instructions](SOUND_GENERATOR.md) to get it running

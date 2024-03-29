# Neural Audio Synthesis with Timbre Conditioned Auto Encoder

TBD

## Web App

The system is deployed as a web app [here](https://share.streamlit.io/thesoundofaiosr/rg_sound_generation/main/app.py)

## Data Preparation

TBD

## Training

Use a `LocalConfig` instance to control architecture and training parameters

```python
from tcae import localconfig, train

conf = localconfig.LocalConfig()

conf.batch_size = 8
conf.simple_encoder = True
conf.simple_decoder = True
conf.save_config()

train.train(conf)
```

## Sound Generator

Deploy a trained model as a `SoundGenerator`. A more complete example can be found [here](deploy.ipynb)

```python
from sound_generator import SoundGenerator


sg = SoundGenerator()

sg.config_path = "/path/to/config"
sg.checkpoint_path = "/path/to/checkpoint.h5"

success, audio = sg.get_prediction({
    "input_pitch": 40,
    "pitch": 40,
    "velocity": 100,
    # A list of sg.conf.num_measures values between 0 and 1
    "heuristic_measures": [0.1] * sg.conf.num_measures,
    # A list of sg.conf.latent_dim values between 0 and 1
    "latent_sample": [0.5] * sg.conf.latent_dim,
    # A list of words describing timbre qualities
    "qualities": ["dark", "soft"],
    # Use this flag if you want to load a good starting point
    # for latent sample and measures. If set to True, it will
    # override latent_sample and measures given in this dict
    "load_preset": True
})

```

Required keys in the input dictionary:

**input_pitch**: Note number to use in decoder input

**pitch**: Note number to use in audio synthesis

**velocity**: Velocity of the note between 25 and 127

**latent_sample**: Values for z input to decoder

**qualities**: Timbre qualities from use speech, used to find initial heuristic configurations

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

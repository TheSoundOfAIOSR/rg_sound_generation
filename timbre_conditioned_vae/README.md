# Timbre Controlled (Variational) Auto Encoder

## Training

TBD

## Dataset

TBD

## Sound Generator

Get audio prediction

```python
from sound_generator import SoundGenerator


sg = SoundGenerator(
    config_path="path/to/conf.json/or/conf.txt",
    checkpoint_path="path/to/checkpoint.h5"
)

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

Expected input dictionary:

```python
{'heuristic_measures': [0.9489624299504555,
                        0.29482239600455784,
                        0.15495734212959966,
                        0.975737484559371,
                        0.865688106222529,
                        0.590016312005441,
                        0.7894165373403864,
                        0.25509719822549504,
                        0.4375296392264817,
                        0.9665697968035902,
                        0.24717908153339518],
 'input_pitch': 40,
 'latent_sample': [0.38556901391643217,
                   0.28356567568667745,
                   0.4047289863627801,
                   0.003463347600569211,
                   0.030082578868353194,
                   0.16126379385761924,
                   0.8532973406004932,
                   0.8333417197580258,
                   0.8821582938686028,
                   0.9666489817879335,
                   0.6531458403907964,
                   0.014581398337647378,
                   0.5181143570334453,
                   0.6431158150445774,
                   0.9205868317010114,
                   0.09085657639464739],
 'pitch': 40,
 'velocity': 100}

```

input_pitch: Note number to use in decoder input

pitch: Note number to use in audio synthesis

velocity: Velocity of the note between 25 and 127

heuristic_measures: List of values for following measures used in decoder in the sequence shown:
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

latent_sample: Values for z input to decoder


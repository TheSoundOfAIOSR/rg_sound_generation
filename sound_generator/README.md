# Sound Generator

End to end inference pipeline for sound generation

## Setup

### Download Models

First create a directory in this folder to store checkpoints, download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1mH8Pqgwxb6nJsx_mCnD9dMBO8qlrmwUq?usp=sharing)

The structure of the module should look like:

```text
sound_generator
    - checkpoints
        - ddsp_generator
        - f0_ld_generator
        - z_generator
    - ddsp_generator
    - f0_ld_generator
    - z_generator
```

### Development

Create a virtual environment and install required packages

```commandline
pip install virtualenv
virtualenv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Now the module can be imported and used

```python
from sound_generator import get_prediction

inputs = {
    'velocity': 75,
    'pitch': 60,
    'source': 'acoustic',
    'qualities': ['bright', 'percussive'],
    'latent_sample': [0.] * 16
}

audio = get_prediction(inputs)
```

### Production

Build and run in a docker container

```commandline
docker build -t <image_tag> .
docker run --rm -it -p 80:80 --name <container_name> <image_tag>
```

Call the endpoint to get prediction

```python
import requests
import json
import numpy as np


URL = 'http://127.0.0.1/sound'

res = requests.post(URL, json={
    'velocity': 75,
    'pitch': 60,
    'source': 'acoustic',
    'qualities': ['bright', 'percussive'],
    'latent_sample': [0.] * 16
})

audio = json.loads(res.text)['audio']
audio = np.squeeze(audio)
```

## Inputs

The `latent_sample` must be a list of 16 floating point values between -7 and +7

The `velocity` can be one of `[25, 50, 75, 100, 127]`

The `pitch` must be between 9 and 120

The `source` can be one of `["acoustic", "electronic", "synthetic"]`

The list `qualities` can be a have a number of qualities from `["bright", "dark", "distortion", "fast_decay", "long_release",
"multiphonic", "nonlinear_env", "percussive", "reverb", "tempo_sync"]`

## To Do

* Both the `z_generator` and `f0_ld_generator` models need to see significant improvemnts
* End to end training with ddsp autoencoder to fine tune the models
* Either categorical outputs for `loudness_db` and `f0_hz` or smoothing to be applied on current predictions or both
* More quality/ descriptor tags are to be used
* Final version should generate only the C note (MIDI pitch 60)

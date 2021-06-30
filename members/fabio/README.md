### Train DDSP on NSynth guitar subset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/train_ddsp_nsynth_guitar.ipynb)

For the dataset generation you have to upload the nsynth audio files to your google drive: `/My Drive/nsynth_guitar/audio`.

### Prepare Partial and Complete tfrecord
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/prepare_partial_complete_tfrecord.ipynb)

For the dataset generation you have to upload the nsynth audio files to your google drive: `/My Drive/nsynth_guitar/train/audio`.

### Generate Harmonic tfrecord dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/generate_harmonic_dataset.ipynb)

### Harmonic dataset demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/harmonic_dataset_demo.ipynb)

### Harmonic autoencoder training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/harmonic_autoencoder_model_training.ipynb)

### GRU Mapping model training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/gru_mapping_model_training.ipynb)

Simple GRU mapping model.  
**Inputs**: `note_number`, `velocity`, `instrument_source`, `qualities`, `z`  
**Outputs**: `f0_scaled`, `ld_scaled`  
The checkpoints are saved into `/My Drive/nsynth_guitar/mapping/checkpoint`.

### Transformer Mapping model training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/transformer_mapping_model_training.ipynb)

Transformer mapping model.  
**Inputs**: `(batches, 285)` - `note_number`, `velocity`, `instrument_source`, `qualities`, `input_z`  
**Outputs**: `(batches, 1000, 18)` - `f0_variation`, `ld_scaled`, `z`  
The checkpoints are saved into `/My Drive/nsynth_guitar/mapping/checkpoint`.

#### Train and Validation sets
```
Epoch 175/500
100/100 [==============================] - 22s 224ms/step - loss: 0.0212 - mean_squared_error: 0.0010 - val_loss: 0.0112 - val_mean_squared_error: 2.5250e-04
```

#### Test set
```
500/500 [==============================] - 46s 91ms/step - loss: 0.0112 - mean_squared_error: 2.5211e-04
```

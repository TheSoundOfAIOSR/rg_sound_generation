### Train DDSP on NSynth guitar subset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/train_ddsp_nsynth_guitar.ipynb)

For the dataset generation you have to upload the nsynth audio files to your google drive: `/My Drive/nsynth_guitar/audio`.

### Prepare Partial and Complete tfrecord
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/prepare_partial_complete_tfrecord.ipynb)

For the dataset generation you have to upload the nsynth audio files to your google drive: `/My Drive/nsynth_guitar/train/audio`.

### Mapping model training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_sound_generation/blob/main/members/fabio/mapping_model_training.ipynb)

Simple GRU mapping model.  
**Inputs**: `note_number`, `velocity`, `instrument_source`, `qualities`, `z`  
**Outputs**: `f0_scaled`, `ld_scaled`  
The checkpoints are saved into `/My Drive/nsynth_guitar/mapping/checkpoint`.

# About
This directory contains two notebooks
 - Exploratory Data Analysis of NSynth dataset
 - Mapping network for f0 and loudness.

## Mapping Network

### LSTM, GRU network
Maps note_number, velocity, instrument_source, qualities, z to f0_scaled and ld_scaled
 -  MSE on test set: 0.0269
 - Tensorboard : [Mapping networks](https://tensorboard.dev/experiment/BA5eBbr1RCGqcgpjtqd0jg/)

### Mapping network using Temporal Convolutional Network

 - MSE on test set:0.0186 
 - Tensorboard: [TCN](https://tensorboard.dev/experiment/N7lWbOeGSW6QF6jQIzdXwg/#scalars)
 
For mapping from note_number, velocity, instrument_source, qualities, z to f0_variance and ld_scaled:[Tensorboard](https://tensorboard.dev/experiment/bSRGcVSkR1Ktk3qgMgfyIQ/#scalars)  

### Mapping network using WaveNet

 - MSE on test set: 3.7678e-04
 - Tensorboard: [WaveNet](https://tensorboard.dev/experiment/RE6d2CvlQfyPJ8MWsPLkqg/)

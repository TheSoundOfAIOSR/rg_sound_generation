# About
This directory contains two notebooks
 - Exploratory Data Analysis of NSynth dataset
 - Mapping network for f0 and loudness.

## Mapping Network

### LSTM, GRU network
Maps note_number, velocity, instrument_source, qualities, z to f0_scaled and ld_scaled
 -  Mean_squared_error on test set: 0.0269
 - Tensorboard : [Mapping networks](https://tensorboard.dev/experiment/BA5eBbr1RCGqcgpjtqd0jg/)

### Mapping network using Temporal Convolutional Network

 - MSE error on test set:0.0186 
 - Tensorboard: [TCN] https://tensorboard.dev/experiment/N7lWbOeGSW6QF6jQIzdXwg/#scalars
 

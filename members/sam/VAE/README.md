# Nsynth Guitar VAE

## Overview
This process creates Mel spectograms of the guitar subset of the Nsynth dataset, then trains a VAE on these spectograms.
In order for the notebooks to run properly, you need to [download](https://magenta.tensorflow.org/datasets/nsynth#files) each Nsynth json/wav file.

## Convert wav to Mel spectograms
Use the notebook **Nsynth-Convert.ipynb** in order to create Mel spectograms from each dataset. 
This saves each spectogram into a local *Download/guitar/* folder, feel free to change this to wherever you'd like. 
If you change the address, please change the references to the load data in the other notebooks as well. 

## Create and Train VAE
Use the notebook **VAE-Nsynth_Guitar.ipynb** in order to create and trian a VAE on the Mel spectograms in the previous step. 
There are options to create a different architecture for the VAE, which correspond to the values in the [evaluations table](https://docs.google.com/spreadsheets/d/1qqgb4VIOz0YEg8GVdhi7kCllb62gEviTza2AdfUPOEA/edit?usp=sharing).
I have saved weights for three architectures, with their file names located in the **Weights** column of the evaluations table. 
These correspond to the best architectures, with latent dimensions: 128 (vae37), 64 (vae35), and 32 (vae30). 
Please download these files to the directory */Downlaods/vae-weights/* or change the function call for *vae.load_weights('Downloads/vae-weights/vae37')*.

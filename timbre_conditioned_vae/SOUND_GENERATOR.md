# Sound Generator App

Requires python >= 3.6

Clone repository

`git clone https://github.com/TheSoundOfAIOSR/rg_sound_generation.git`

Go to the right directory

`cd rg_sound_generation\timbre_conditioned_vae`

Create virtual environment

`python -m venv env`

Activate the environment

`env\Scripts\activate`

Install required packages

`pip install -r requirements.txt`

Download the model

`curl https://osr-tsoai.s3.amazonaws.com/mt_5/model.h5 -o deployed/model.h5`

Run the server

`streamlit run app.py`

Go to http://localhost:8501 - the app should be running

import streamlit as st
import numpy as np
import soundfile as sf
from time import time
from loguru import logger
from sound_generator import SoundGenerator


@st.cache(allow_output_mutation=True)
def load_sound_generator():
    return SoundGenerator(
        config_path="deployed/conf.json",
        checkpoint_path="deployed/model.h5"
    )


measure_max_val = 149
z_max_val = 99
default_z = 50

st.title("Sound Generator")

z1 = st.sidebar.slider("z1", min_value=0, max_value=z_max_val, value=default_z)
z2 = st.sidebar.slider("z2", min_value=0, max_value=z_max_val, value=default_z)
z3 = st.sidebar.slider("z3", min_value=0, max_value=z_max_val, value=default_z)
z4 = st.sidebar.slider("z4", min_value=0, max_value=z_max_val, value=default_z)
z5 = st.sidebar.slider("z5", min_value=0, max_value=z_max_val, value=default_z)

inharmonicity = st.sidebar.slider("inharmonicity", min_value=0, max_value=measure_max_val)
even_odd = st.sidebar.slider("even_odd", min_value=0, max_value=measure_max_val)
sparse_rich = st.sidebar.slider("sparse_rich", min_value=0, max_value=measure_max_val)
attack_rms = st.sidebar.slider("attack_rms", min_value=0, max_value=measure_max_val)
decay_rms = st.sidebar.slider("decay_rms", min_value=0, max_value=measure_max_val)
attack_time = st.sidebar.slider("attack_time", min_value=0, max_value=measure_max_val)
decay_time = st.sidebar.slider("decay_time", min_value=0, max_value=measure_max_val)
bass = st.sidebar.slider("bass", min_value=0, max_value=measure_max_val)
mid = st.sidebar.slider("mid", min_value=0, max_value=measure_max_val)
high_mid = st.sidebar.slider("high_mid", min_value=0, max_value=measure_max_val)
high = st.sidebar.slider("high", min_value=0, max_value=measure_max_val)

input_pitch = st.slider("input_pitch", min_value=40, max_value=88, value=60)
output_pitch = st.slider("output_pitch", min_value=40, max_value=88, value=60)
velocity = st.slider("velocity", min_value=25, max_value=125, value=75, step=25)


if st.button("Generate"):
    z = [z / 100 for z in [z1, z2, z3, z4, z5]]
    measures = [m / 100 for m in [
        inharmonicity, even_odd, sparse_rich, attack_rms,
        decay_rms, attack_time, decay_time, bass, mid,
        high_mid, high
    ]]
    data = {
        "input_pitch": input_pitch,
        "pitch": output_pitch,
        "velocity": 100,
        "heuristic_measures": measures,
        "latent_sample": z
    }

    sg = load_sound_generator()
    start = time()
    success, audio = sg.get_prediction(data)
    logger.info(f"Time taken for prediction + generation: {time() - start: .3} seconds")
    if success:
        audio = np.array(audio) / np.max(np.abs(audio))
        sf.write("temp.wav", audio, samplerate=sg.conf.sample_rate)

    st.subheader("Audio")
    st.audio("temp.wav", format="audio/wav")

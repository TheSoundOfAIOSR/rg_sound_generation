import streamlit as st
import numpy as np
import random
import soundfile as sf
from time import time
from loguru import logger
from sound_generator import SoundGenerator
from tcae.localconfig import LocalConfig


@st.cache(allow_output_mutation=True)
def load_sound_generator():
    return SoundGenerator()


@st.cache(allow_output_mutation=True)
def get_known_zs():
    _known_zs = []
    with open("latent_samples.csv", "r") as f:
        _data = f.read().splitlines()
        _data = _data[1:]
        for row in _data:
            _known_zs.append([int(float(x) * 100) for x in row.split(",")])
    return _known_zs


known_zs = get_known_zs()
measure_max_val = 100
z_max_val = 100
default_z = [50] * 2
default_m = [50] * 11

if "latent_sample" not in st.session_state:
    st.session_state["latent_sample"] = default_z
else:
    default_z = st.session_state["latent_sample"]


col1, col2, col3 = st.beta_columns(3)


col1.title("Generator")
col2.subheader("Latent Sample")
col3.subheader("Measures")


input_pitch = col1.slider("input_pitch", min_value=40, max_value=88, value=60)
output_pitch = col1.slider("output_pitch", min_value=40, max_value=88, value=60)
velocity = col1.slider("velocity", min_value=25, max_value=125, value=75, step=25)


# if col2.button("Reset Z"):
#     st.session_state.clear()
#     default_z = [50] * 2
#
# if col2.button("Get Suggested Z"):
#     default_z = random.choice(known_zs)
#     logger.info("Got suggested Z, updating session state")
#     st.session_state["latent_sample"] = default_z

z1 = col2.slider("z1", min_value=0, max_value=z_max_val, value=default_z[0])
z2 = col2.slider("z2", min_value=0, max_value=z_max_val, value=default_z[1])


inharmonic = col3.slider("inharmonic", min_value=0, max_value=measure_max_val, value=default_m[0])
even_odd = col3.slider("even_odd", min_value=0, max_value=measure_max_val, value=default_m[1])
sparse_rich = col3.slider("sparse_rich", min_value=0, max_value=measure_max_val, value=default_m[2])
attack_rms = col3.slider("attack_rms", min_value=0, max_value=measure_max_val, value=default_m[3])
decay_rms = col3.slider("decay_rms", min_value=0, max_value=measure_max_val, value=default_m[4])
attack_time = col3.slider("attack_time", min_value=0, max_value=measure_max_val, value=default_m[5])
decay_time = col3.slider("decay_time", min_value=0, max_value=measure_max_val, value=default_m[6])
bass = col3.slider("bass", min_value=0, max_value=measure_max_val, value=default_m[7])
mid = col3.slider("mid", min_value=0, max_value=measure_max_val, value=default_m[8])
high_mid = col3.slider("high_mid", min_value=0, max_value=measure_max_val, value=default_m[9])
high = col3.slider("high", min_value=0, max_value=measure_max_val, value=default_m[10])


if col1.button("Generate"):
    sg = load_sound_generator()

    z = [z / z_max_val for z in [z1, z2]]
    measures = dict((m, 2.0 * (eval(m) / measure_max_val - 0.5)) for m in sg.conf.data_handler.measure_names)
    # measures = sg.conf.data_handler.measures_mapping(measures)

    conf = LocalConfig()

    note_index = input_pitch - conf.starting_midi_pitch
    velocity_index = velocity // 25 - 1

    measures = conf.data_handler.shift_measures_mean(
        measures, note_index, velocity_index)

    data = {
        "input_pitch": input_pitch,
        "pitch": output_pitch,
        "velocity": velocity,
        "heuristic_measures": list(measures.values()),
        "latent_sample": z
    }

    start = time()
    success, audio = sg.get_prediction(data)
    logger.info(f"Time taken for prediction + generation: {time() - start: .3} seconds")
    if success:
        audio = np.array(audio) / np.max(np.abs(audio))
        sf.write("temp.wav", audio, samplerate=sg.conf.sample_rate)

    col1.subheader("Audio")
    col1.audio("temp.wav", format="audio/wav")

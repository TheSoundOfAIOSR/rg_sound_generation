import os
import streamlit as st
import numpy as np
import soundfile as sf
import pickle
from uuid import uuid4
from time import time
from loguru import logger
from sound_generator import SoundGenerator
from decoder_inputs_export import compute_encoding


tmp_dir = os.path.join(os.getcwd(), "tmp")

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)


@st.cache(allow_output_mutation=True)
def load_sound_generator():
    return SoundGenerator()


@st.cache(allow_output_mutation=True)
def load_decoded_values():
    with open('decoder_inputs.pickle', 'rb') as f:
        decoded_inputs = pickle.load(f)
    return decoded_inputs


@st.cache(allow_output_mutation=True)
def get_known_zs():
    _known_zs = []
    with open("latent_samples.csv", "r") as f:
        _data = f.read().splitlines()
        _data = _data[1:]
        for row in _data:
            _known_zs.append([int(float(x) * 100) for x in row.split(",")])
    return _known_zs


def measure_transform(measure_value, measure_mean):
    measure_value = 2.0 * measure_value - 1.0

    if measure_value >= 0.0:
        measure_value = measure_mean + measure_value * (1.0 - measure_mean)
    else:
        measure_value = (1.0 + measure_value) * measure_mean
    return measure_value


def inverse_measure_transform(measure_value, measure_mean):
    if measure_value >= measure_mean:
        measure_value = (measure_value - measure_mean) / (1.0 - measure_mean)
    else:
        measure_value = (measure_value - measure_mean) / measure_mean

    measure_value = (measure_value + 1.0) / 2.0
    return measure_value


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


sg = load_sound_generator()
decoded_values = load_decoded_values()

col1.subheader("Harmonic")
col2.subheader("Temporal")
col3.subheader("Frequency")

st.sidebar.subheader("Global Parameters")

instrument_id = st.sidebar.selectbox("Instrument ID", options=list(range(0, sg.conf.num_instruments)))
output_pitch = st.sidebar.slider("midi_note_number", min_value=40, max_value=88, value=60)
velocity = st.sidebar.slider("velocity", min_value=25, max_value=127, value=75, step=1)
input_pitch = st.sidebar.slider("conditioning_note_number", min_value=40, max_value=88, value=60)

note_index = input_pitch - sg.conf.starting_midi_pitch
velocity_index = velocity // 25 - 1

measures_mean = sg.conf.data_handler.get_measures_mean(
    note_index, velocity_index)

decoder_index = compute_encoding(note_index, velocity_index, instrument_id, sg.conf)

if decoder_index < len(decoded_values):
    decoder_value = decoded_values[decoder_index]
    if decoder_value["z"] is not None and decoder_value["measures"] is not None:
        default_z = [int(x * z_max_val) for x in decoder_value["z"].numpy()[0]]
        default_m = [int(inverse_measure_transform(v, measures_mean[k]) * measure_max_val)
                     for k, v in zip(sg.conf.data_handler.measure_names, decoder_value["measures"].numpy()[0])]
else:
    logger.warning(f"decoder index {decoder_index} not found in decoded values")


st.sidebar.subheader("Latent Sample")

z1 = st.sidebar.slider("z1", min_value=0, max_value=z_max_val, value=default_z[0])
z2 = st.sidebar.slider("z2", min_value=0, max_value=z_max_val, value=default_z[1])


inharmonic = col1.slider("inharmonic", min_value=0, max_value=measure_max_val, value=default_m[0])
even_odd = col1.slider("even_odd", min_value=0, max_value=measure_max_val, value=default_m[1])
sparse_rich = col1.slider("sparse_rich", min_value=0, max_value=measure_max_val, value=default_m[2])
attack_rms = col2.slider("attack_rms", min_value=0, max_value=measure_max_val, value=default_m[3])
decay_rms = col2.slider("decay_rms", min_value=0, max_value=measure_max_val, value=default_m[4])
attack_time = col2.slider("attack_time", min_value=0, max_value=measure_max_val, value=default_m[5])
decay_time = col2.slider("decay_time", min_value=0, max_value=measure_max_val, value=default_m[6])
bass = col3.slider("bass", min_value=0, max_value=measure_max_val, value=default_m[7])
mid = col3.slider("mid", min_value=0, max_value=measure_max_val, value=default_m[8])
high_mid = col3.slider("high_mid", min_value=0, max_value=measure_max_val, value=default_m[9])
high = col3.slider("high", min_value=0, max_value=measure_max_val, value=default_m[10])


if st.sidebar.button("Generate"):
    z = [z / z_max_val for z in [z1, z2]]
    measures = dict((m, eval(m) / measure_max_val) for m in sg.conf.data_handler.measure_names)
    measures = dict((k, measure_transform(v, measures_mean[k])) for k, v in measures.items())

    data = {
        "input_pitch": input_pitch,
        "pitch": output_pitch,
        "velocity": velocity,
        "heuristic_measures": list(measures.values()),
        "latent_sample": z,
        "instrument_id": instrument_id
    }

    start = time()
    success, audio = sg.get_prediction(data)
    logger.info(f"Time taken for prediction + generation: {time() - start: .3} seconds")

    if success:
        tmp_file_name = f"{uuid4()}.wav"
        tmp_file_path = os.path.join(tmp_dir, tmp_file_name)
        audio = np.squeeze(np.array(audio) / np.max(np.abs(audio)))
        sf.write(tmp_file_path, audio, samplerate=sg.conf.sample_rate)

        st.sidebar.subheader("Audio")
        st.sidebar.audio(tmp_file_path, format="audio/wav")


st.subheader("Sound Generator")
st.text("This app can be used to generate (hopefully) usable one shot guitar samples")
st.text("Description of parameters comes here")

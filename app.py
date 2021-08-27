import streamlit as st
import numpy as np
import io
from scipy.io import wavfile
from time import time
from loguru import logger
from sound_generator import SoundGenerator


@st.cache(allow_output_mutation=True)
def load_sound_generator():
    return SoundGenerator()


def get_audio_prediction(z1, z2, input_pitch, output_pitch, velocity, instrument_id):
    logger.info(f"First z for sanity check {z1}")
    z = [z / z_max_val for z in [z1, z2]]
    measures = dict((m, eval(m) / measure_max_val) for m in sg.conf.data_handler.measure_names)
    # measures = dict((k, measure_transform(v, measures_mean[k])) for k, v in measures.items())

    # Load preset is always false because we are manually loading presets based on user input
    # in the streamlit app below
    data = {
        "input_pitch": input_pitch,
        "pitch": output_pitch,
        "velocity": velocity,
        "heuristic_measures": list(measures.values()),
        "latent_sample": z,
        "instrument_id": instrument_id,
        "load_preset": False
    }

    start = time()
    result = sg.get_prediction(data)
    success = result.get("success")
    audio = result.get("audio")
    logger.info(f"Time taken for prediction + generation: {time() - start: .3} seconds")
    return success, audio


def get_audio_bytes(audio):
    audio_norm = np.squeeze(np.array(audio) / np.max(np.abs(audio)))

    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    wavfile.write(byte_io, sg.conf.sample_rate, audio_norm.astype("float32"))
    result_bytes = byte_io.read()
    return result_bytes


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

col1.subheader("Harmonic")
col2.subheader("Temporal")
col3.subheader("Frequency")

st.sidebar.subheader("Global Parameters")

instrument_id = st.sidebar.selectbox("Instrument ID", options=list(range(0, sg.conf.num_instruments)))
prev_id = -1
if "instrument_id" in st.session_state:
    prev_id = st.session_state["instrument_id"]

output_pitch = st.sidebar.slider("midi_note_number", min_value=40, max_value=88, value=60)
velocity = st.sidebar.slider("velocity", min_value=25, max_value=127, value=75, step=1)
input_pitch = output_pitch


if instrument_id != prev_id:
    logger.info(f"Instrument id is changed from {prev_id} to {instrument_id}")
    st.session_state["instrument_id"] = instrument_id

    measures_mean, note_index, velocity_index = sg.get_measures_mean(
        input_pitch=input_pitch, velocity=velocity
    )
    success, values = sg.load_preset_fn(measures_mean, note_index, velocity_index, instrument_id)

    if success:
        default_z, default_m = values
        default_z = [int(val * z_max_val) for val in default_z]
        default_m = [int(val * measure_max_val) for val in default_m]
        st.session_state["default_z"] = default_z
        st.session_state["default_m"] = default_m
        logger.info("Default z and m are updated to decoder values in session state")
    else:
        logger.info("Could not successfully load preset")
else:
    logger.info(f"Instrument id {instrument_id} did not change")
    if "default_z" in st.session_state:
        default_z = st.session_state["default_z"]
    if "default_m" in st.session_state:
        default_m = st.session_state["default_m"]


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
    success, audio = get_audio_prediction(z1, z2, input_pitch, output_pitch, velocity, instrument_id)

    if success:
        result_bytes = get_audio_bytes(audio)
        st.sidebar.subheader("Audio")
        st.sidebar.audio(result_bytes, format="audio/wav")


st.subheader("Sound Generator")
st.text("Note: If you change Instrument ID, values of Z and measures are reset")

# Todo: Let users export a complete sample pack
# st.text("Once you're happy with a configuration, you can export the sample pack")
#
# pack_name = st.text_input("Enter a name for your pack")
#
#
# if st.button("Export Sample Pack"):
#     folder_name = str(shortuuid.uuid())
#     folder_path = DOWNLOADS_PATH / folder_name
#
#     os.mkdir(folder_path)
#
#     pack_name = [c for c in pack_name.lower() if c in string.ascii_lowercase]
#     pack_name = "".join(pack_name)
#     pack_name = pack_name if pack_name != "" else os.path.basename(folder_path)
#
#     progress_bar = st.progress(0)
#     items_count = sg.conf.num_pitches * sg.conf.num_velocities
#
#     for i, n in enumerate(range(40, 45)):
#         for j, v in enumerate([25, 50, 75, 100, 125]):
#             success, audio = get_audio_prediction(
#                 z1, z2, n, n, v, instrument_id
#             )
#             if success:
#                 audio_bytes = get_audio_bytes(audio)
#                 file_path = folder_path / f"{n}_v{v}_{pack_name}.wav"
#
#                 with open(file_path, "wb") as f:
#                     f.write(audio_bytes)
#             progress_bar.progress((i * sg.conf.num_velocities + j) / items_count)
#
#     st.text("Creating archive.. Wait for the download link:")
#
#     archive_path = DOWNLOADS_PATH / f"{os.path.basename(folder_path)}"
#     shutil.make_archive(archive_path, "zip", folder_path)
#
#     logger.info(f"Archived saved at {archive_path}.zip")
#
#     st.markdown(f"<a href='{archive_path}.zip'>Right Click and Save As</a>", unsafe_allow_html=True)

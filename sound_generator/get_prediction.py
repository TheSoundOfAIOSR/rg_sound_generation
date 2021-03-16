"""
Sound Generator
"""
import os
import sound_generator

from sound_generator.z_generator import ZGenerator, ZDataProcessor
from sound_generator.f0_ld_generator import F0LoudnessGenerator, F0LoudnessDataProcessor
from sound_generator.ddsp_generator import DDSPGenerator, DDSPDataProcessor


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
base_path = os.path.dirname(os.path.abspath(sound_generator.__file__))
print(base_path)
# Create Data Processors
z_data_processor = ZDataProcessor()
f0_ld_data_processor = F0LoudnessDataProcessor()
ddsp_data_processor = DDSPDataProcessor()

# Create Models
z_model = ZGenerator(checkpoint_path=os.path.join(base_path, 'checkpoints/z_generator/cp.ckpt'))
f0_ld_model = F0LoudnessGenerator(checkpoint_path=os.path.join(base_path, 'checkpoints/f0_ld_generator/cp.ckpt'))
ddsp_model = DDSPGenerator(
    checkpoint_path=os.path.join(base_path, 'checkpoints/ddsp_generator'),
    gin_config=os.path.join(base_path, 'checkpoints/ddsp_generator/operative_config-30000.gin')
)


def get_prediction(inputs):
    z_inputs = z_data_processor.process(inputs)
    z_outputs = z_model.predict(z_inputs)

    f0_ld_inputs = f0_ld_data_processor.process({
        'z_inputs': z_inputs,
        'z_outputs': z_outputs
    })
    f0_ld_outputs = f0_ld_model.predict(f0_ld_inputs)

    ddsp_inputs = ddsp_data_processor.process({
        'f0_ld_inputs': f0_ld_inputs,
        'f0_ld_outputs': f0_ld_outputs
    })
    audio = ddsp_model.predict(ddsp_inputs)
    return audio

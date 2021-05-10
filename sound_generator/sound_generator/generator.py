import os
from typing import Dict, Any
from loguru import logger
from .z_generator import ZGenerator, ZDataProcessor
from .f0_ld_generator import F0LoudnessGenerator, F0LoudnessDataProcessor
from .ddsp_generator import DDSPGenerator, DDSPDataProcessor


class SoundGenerator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating SoundGenerator instance. "
                        "This should run exactly once during the lifetime of the module")
            cls._instance = super(SoundGenerator, cls).__new__(cls)
        return cls._instance

    def build(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.z_model_path = os.path.join(self.base_path,
                                         'checkpoints/z_generator/cp.ckpt')
        self.f0_ld_model_path = os.path.join(self.base_path,
                                             'checkpoints/f0_ld_generator/cp.ckpt')
        self.ddsp_model_path = os.path.join(self.base_path,
                                            'checkpoints/ddsp_generator')
        self.ddsp_gin_path = os.path.join(self.base_path,
                                          'checkpoints/ddsp_generator/operative_config-30000.gin')
        self._models_loaded = False

        self.z_model = None
        self.f0_ld_model = None
        self.ddsp_model = None

        self.z_data_processor = ZDataProcessor()
        self.f0_ld_data_processor = F0LoudnessDataProcessor()
        self.ddsp_data_processor = DDSPDataProcessor()

    @property
    def models_loaded(self):
        return self._models_loaded

    def load_models(self):
        try:
            logger.info("Loading models")
            logger.info("Verifying model checkpoints")

            assert os.path.isdir(os.path.dirname(self.z_model_path)), \
                f"There is no model checkpoint at {self.z_model_path}"
            assert os.path.isdir(os.path.dirname(self.f0_ld_model_path)), \
                f"There is no model checkpoint at {self.f0_ld_model_path}"
            assert os.path.isdir(os.path.dirname(self.ddsp_model_path)), \
                f"There is no model checkpoint at {self.ddsp_model_path}"

            # Load Models
            self.z_model = ZGenerator(checkpoint_path=self.z_model_path)
            self.f0_ld_model = F0LoudnessGenerator(checkpoint_path=self.f0_ld_model_path)
            self.ddsp_model = DDSPGenerator(
                checkpoint_path=self.ddsp_model_path,
                gin_config=self.ddsp_gin_path
            )
            self._models_loaded = True
            logger.info("All models loaded")
        except Exception as e:
            logger.error(e)

    def predict(self, inputs: Dict) -> Any:
        if not self._models_loaded:
            logger.error("Models are not loaded yet. Have you "
                         "downloaded the pretrained checkpoints?")
            return None

        logger.info("Processing input for z_generator")
        z_inputs = self.z_data_processor.process(inputs)
        logger.info("Getting output from z_generator")
        z_outputs = self.z_model.predict(z_inputs)

        logger.info("Processing input for f0_ld_generator")
        f0_ld_inputs = self.f0_ld_data_processor.process({
            'z_inputs': z_inputs,
            'z_outputs': z_outputs
        })
        logger.info("Getting output from f0_ld_generator")
        f0_ld_outputs = self.f0_ld_model.predict(f0_ld_inputs)

        logger.info("Processing input for ddsp_generator")
        ddsp_inputs = self.ddsp_data_processor.process({
            'f0_ld_inputs': f0_ld_inputs,
            'f0_ld_outputs': f0_ld_outputs
        }, target_pitch=self.z_data_processor.pitch)
        logger.info("Getting output from ddsp_generator")
        audio = self.ddsp_model.predict(ddsp_inputs)
        return audio.numpy()

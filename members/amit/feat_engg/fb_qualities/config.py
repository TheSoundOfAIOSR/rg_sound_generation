
class Config:
    _instance = None

    sample_rate = 16000
    frame_length = 1024
    frame_step = 256
    fft_length = 256

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance


config = Config()

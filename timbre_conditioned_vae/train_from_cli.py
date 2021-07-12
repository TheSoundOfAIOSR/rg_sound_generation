from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()

    conf.dataset_dir = "/content/drive/MyDrive/the_sound_of_ai/new_data"
    conf.checkpoints_dir = "/content/drive/MyDrive/the_sound_of_ai/new_data/checkpoints"
    conf.use_encoder = False
    conf.decoder_type = "cnn"
    conf.batch_size = 8
    conf.learning_rate = 2e-3
    conf.model_name = "cnn_decoder_heuristics"
    conf.step_log_interval = 10
    conf.reconstruction_weight = 1.
    conf.save_config()

    train.train(conf)

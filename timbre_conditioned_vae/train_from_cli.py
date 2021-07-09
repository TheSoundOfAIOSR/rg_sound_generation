from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()

    conf.batch_size = 1
    conf.latent_dim = 16
    conf.kl_weight = 1.
    conf.epochs = 1
    conf.use_encoder = False
    conf.decoder_type = "rnn"
    conf.step_log_interval = 2

    train.train(conf)

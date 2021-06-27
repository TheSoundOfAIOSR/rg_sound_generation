from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()

    conf.batch_size = 4
    conf.latent_dim = 64
    conf.freq_loss_weight = 10
    conf.best_loss = 1.1256
    conf.kl_weight = 0.2
    conf.epochs = 1000

    train.train(conf)

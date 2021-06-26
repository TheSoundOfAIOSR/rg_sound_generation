from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()
    train.train(conf)

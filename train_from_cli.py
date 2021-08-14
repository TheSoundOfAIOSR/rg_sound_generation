from tcae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig("data_handler")

    conf.use_encoder = True
    conf.use_heuristics = True
    conf.latent_dim = 16

    conf.batch_size = 2
    conf.model_name = "test"
    conf.print_model_summary = False

    conf.simple_encoder = True
    conf.simple_decoder = True
    conf.using_categorical = False

    conf.data_handler.update_losses_weights(measures=1.)

    train.train(conf)

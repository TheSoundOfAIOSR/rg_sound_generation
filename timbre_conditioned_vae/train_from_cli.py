import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()

    conf.use_encoder = True
    conf.is_variational = False
    conf.decoder_type = "cnn"
    conf.batch_size = 4
    conf.learning_rate = 2e-3
    conf.model_name = "test"
    conf.step_log_interval = 1
    conf.data_handler.f0_weight_type = "mag_max_pool"
    conf.data_handler.mag_loss_type = "mse"

    # conf.save_config()

    train.train(conf)

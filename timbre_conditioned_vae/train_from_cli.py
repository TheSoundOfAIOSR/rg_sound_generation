import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()
    # conf.load_config_from_file("checkpoints/Default_test.json")
    conf.use_encoder = True
    conf.use_max_pool = False
    conf.is_variational = False
    conf.add_z_to_decoder_blocks = False
    conf.encoder_type = "1d"
    conf.decoder_type = "cnn"
    conf.latent_dim = 1024
    # Training
    conf.batch_size = 2
    conf.learning_rate = 2e-4
    conf.lr_factor = 0.5
    conf.lr_plateau = 4
    conf.model_name = "test"
    conf.early_stopping = 10
    # Data Handler
    conf.data_handler.mag_loss_type = "l2_db"
    conf.data_handler.mag_scale_fn = "exp_sigmoid"

    conf.num_train_steps = 5
    conf.num_valid_steps = 3
    conf.epochs = 2

    # conf.save_config()

    train.train(conf)

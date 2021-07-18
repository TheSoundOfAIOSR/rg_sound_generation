import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig()
    # conf.load_config_from_file("checkpoints/Default_test.json")
    conf.use_encoder = True
    conf.use_max_pool = False
    conf.is_variational = True
    conf.use_heuristics = False
    conf.add_z_to_decoder_blocks = False
    conf.deep_decoder = True
    conf.encoder_type = "2d"
    conf.decoder_type = "cnn"
    conf.latent_dim = 64
    # Training
    conf.batch_size = 2
    conf.learning_rate = 2e-4
    conf.lr_factor = 0.5
    conf.lr_plateau = 4
    conf.model_name = "test"
    conf.early_stopping = 10
    conf.print_model_summary = True
    conf.check_decoder_hidden_dim = False
    # Data Handler
    conf.data_handler.mag_loss_type = "l2_db"
    conf.data_handler.mag_scale_fn = "exp_sigmoid"

    conf.num_train_steps = 5
    conf.num_valid_steps = 3
    conf.epochs = 2

    # conf.save_config()

    train.train(conf)

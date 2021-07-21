import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig("data_handler")

    conf.use_encoder = True
    conf.use_heuristics = False
    conf.latent_dim = 64
    # Training
    conf.batch_size = 2
    conf.learning_rate = 2e-4
    conf.lr_factor = 0.5
    conf.lr_plateau = 4
    conf.model_name = "test"
    conf.early_stopping = 10
    conf.print_model_summary = True
    # Data Handler

    conf.num_train_steps = 5
    conf.num_valid_steps = 3
    conf.epochs = 1

    # conf.save_config()
    # conf.load_config_from_file("checkpoints/Default_test.json")

    train.train(conf)

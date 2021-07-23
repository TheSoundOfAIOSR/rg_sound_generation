import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import localconfig, train


if __name__ == "__main__":
    conf = localconfig.LocalConfig("data_handler")

    conf.use_encoder = True
    conf.use_heuristics = True
    conf.latent_dim = 16
    # Training
    conf.batch_size = 4
    conf.learning_rate = 2e-3
    conf.lr_factor = 0.5
    conf.lr_plateau = 4
    conf.model_name = "test"
    conf.early_stopping = 10
    conf.print_model_summary = True
    # Data Handler

    conf.num_train_steps = 2
    conf.num_valid_steps = 2
    conf.epochs = 1
    conf.simple_encoder = True
    conf.simple_decoder = True

    conf.mt_outputs["mag_env"]["enabled"] = True
    conf.mt_outputs["h_freq_shifts"]["enabled"] = True
    conf.mt_outputs["f0_shifts"]["enabled"] = True
    conf.mt_outputs["h_mag_dist"]["enabled"] = True
    conf.mt_outputs["h_phase_diff"]["enabled"] = False

    # conf.data_handler.losses_weights["mag_env_loss"] = 0
    # conf.save_config()
    # conf.load_config_from_file("checkpoints/Default_test.json")

    train.train(conf)

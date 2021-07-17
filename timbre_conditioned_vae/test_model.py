from tcvae import model, localconfig


c = localconfig.LocalConfig()
c.use_encoder = True
c.use_max_pool = False
c.is_variational = False
c.use_lstm_in_encoder = True
c.add_z_to_decoder_blocks = False
c.latent_dim = 1024
m = model.get_model_from_config(c)

print(m.summary())

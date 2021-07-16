from tcvae import localconfig, model


c = localconfig.LocalConfig()
c.use_encoder = True
c.latent_dim = 1024
c.add_z_to_decoder_blocks = False

m = model.create_decoder(c)

print(m.summary())

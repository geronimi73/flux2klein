

def add_noise(latent, noise, timestep):
  "Add given noise at given level (`timestep`) to latent"
  return (1 - timestep) * latent + timestep * noise # (1-noise_level) * latent + noise_level * noise   

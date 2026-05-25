import torch

def get_schedule(num_steps, rho=5):
  "Karras et al schedule for sigma_max = 1 and sigma_min = 0"
  return torch.linspace(1, 0, num_steps + 1) ** (1/rho)

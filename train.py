import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from itertools import cycle
from einops import rearrange

from flux2klein import (
  load_transformer_flux2klein4base,
  load_ae,
  ae_encode,
  ae_decode,
  prc_txt, 
  prc_img, 
)
from core.images import pil_cat

def train(
  device = "cuda" if torch.cuda.is_available() else "mps",
  dtype = torch.bfloat16,
  dataset = "g-ronimo/masked_background_v6",
  lr = 1e-4,
  steps = 4001,
  seed = 42,
  mock = True
):
  """
    Trains flux2klein 4b base as an object remover. 
    No guidance yet
    WIP! 
  """
  prompt_tok = torch.load("cache/prompt_remove.pt", map_location="cpu").to(device)
  transformer = load_transformer_flux2klein4base(mock=mock).to(dtype).to(device)
  ae = load_ae(mock=mock).to(dtype).to(device)
  ds = load_dataset(dataset)["train"]
  data_sampler = cycle(torch.utils.data.RandomSampler(ds, generator=torch.manual_seed(seed)))
  optimizer = (
    bnb.optim.AdamW8bit(transformer.parameters(), lr=lr) if device == "cuda" else
    torch.optim.AdamW(transformer.parameters(), lr=lr) 
  )
  img_eval, _ = preprocess_sample(load_dataset(dataset)["eval"][0])

  for step in range(steps):
    transformer.train()

    # Sample input (=with masked area) and target image
    img_in, img_target = preprocess_sample(ds[next(data_sampler)])
    if step == 0:
      log_first_sample(img_in, img_target)
    img_in_size = img_in.size
    img_in = ae_encode(ae, img_in).squeeze()
    img_target = ae_encode(ae, img_target).squeeze()

    # Choose noise level
    timestep = get_rnd_timestep(1).to(device).to(dtype)

    # Add noise to input img
    noise = torch.randn([128, img_in_size[1]//16, img_in_size[0]//16], dtype=dtype, device=device)
    img_in = add_noise(img_in, noise, timestep)

    # Add IDs to prompt, noise, input image
    txt, txt_ids = prc_txt(prompt_tok)
    img, img_ids = prc_img(noise)
    img_ref, img_ref_ids = prc_img(img_in.squeeze(), t_coord=torch.tensor([10], dtype=torch.int64))

    # For loss: flatten noise and clean target 
    noise_flat, _ = prc_img(noise)
    img_target_flat, _ = prc_img(img_target)

    # Add batch dimension 
    txt, txt_ids = txt[None,], txt_ids[None,]
    img, img_ids = img[None,], img_ids[None,]
    img_ref, img_ref_ids = img_ref[None,], img_ref_ids[None,]
    noise_flat, img_target_flat = noise_flat[None,], img_target_flat[None,]

    pred = transformer.forward(
      x =     torch.cat([img, img_ref], dim=1), 
      x_ids = torch.cat([img_ids, img_ref_ids], dim=1),
      ctx = txt, 
      ctx_ids = txt_ids,
      timesteps = timestep,
      guidance = None
    )
    pred, _ = pred.split(pred.shape[1] // 2, dim=1) 

    loss = F.mse_loss(pred, noise_flat - img_target_flat)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.detach().cpu().item()
    print(f"Step {step} loss: {loss:.2f}, grad_norm: {grad_norm:.2f}")

    if step % 50 == 0:
      eval_step(step, transformer, ae, prompt_tok, img_eval)

def log_first_sample(img_in, img_target):
  print("First sample image saved. Size: ", img_in.size, f"(mode {img_in.mode})")
  pil_cat(img_in, img_target).save("sample_zero.jpg")

def eval_step(step, transformer, ae, prompt_tok, image):
  transformer.eval()

  image_out_fn = f"eval-{step}_output.jpg"
  image_out = img2img(transformer, ae, prompt_tok, image)
  pil_cat(image, image_out).save(image_out_fn)
  print(f"Eval at step {step}. Output written to {image_out_fn}")

  torch.cuda.empty_cache()

def img2img(
  transformer,
  ae,
  prompt_tok,
  img_ref,
  num_steps = 50,
  seed = 42,
  guidance = 4,
  device = "cuda" if torch.cuda.is_available() else "mps",
  dtype = torch.bfloat16,
  output_fn = "output.jpg"
  ):
  # 99% same as text2img(), changes as comments
  if seed is not None:
    torch.manual_seed(seed)

  img_w, img_h = img_ref.size
  noise = torch.randn([128, img_h//16, img_w//16], dtype=dtype, device=device)
  img_ref = ae_encode(ae, img_ref).squeeze()
  
  # Add IDs to prompt, noise, input image
  txt, txt_ids = prc_txt(prompt_tok)
  img, img_ids = prc_img(noise)
  img_ref, img_ref_ids = prc_img(img_ref.squeeze(), t_coord=torch.tensor([10], dtype=torch.int64))

  # Add batch dimension 
  txt, txt_ids = txt[None,], txt_ids[None,]
  img, img_ids = img[None,], img_ids[None,]
  img_ref, img_ref_ids = img_ref[None,], img_ref_ids[None,]

  timesteps = get_schedule(num_steps)
  # guidance_vec = torch.full((img.shape[0],), 1.0, device=device, dtype=dtype)

  for step, (t_curr, t_next) in enumerate(
    tqdm(zip(timesteps, timesteps[1:]), desc="Denoising", total=num_steps)
    ):
    timesteps_vec = torch.full((img.shape[0],), t_curr, device=device, dtype=dtype)    

    with torch.no_grad():
      pred = transformer.forward(
        # x: Concatenated img(=noise)+ref imgs; "id" coords encoded with different `t` dim for each img. that's the most important point
        x =     torch.cat([img, img_ref], dim=1), 
        x_ids = torch.cat([img_ids, img_ref_ids], dim=1),
        ctx = txt, ctx_ids = txt_ids,
        timesteps = timesteps_vec,
        guidance = None
      )
      # model returns [img + img_refs] -> strip img_refs
      pred = pred[:, :img.size(1)]
    img = img + (t_next - t_curr) * pred

  # unflatten tensor; linear -> 2d  
  return ae_decode(
    ae,
    rearrange(img, "1 (h w) c -> 1 c h w", h=img_h//16)
  )

def preprocess_sample(sample, resize_to=512, patch_size=16):
  "Load single image input and target from dataset"
  img_target, mask = sample["image"], sample["mask"]
  img_target = img_target.convert("RGB")
  img_in = img_target.copy()
  img_in.paste((255,)*3, mask=mask.convert("L"))  # paste white where mask is white

  # Resize to closest mult. of patch size
  imgs_preprocessed = []
  for img in [img_in, img_target]:
    img = transforms.Resize(resize_to)(img)
    img = transforms.CenterCrop(tuple(x//patch_size*patch_size for x in (img.height, img.width)))(img)
    imgs_preprocessed.append(img)

  return imgs_preprocessed

def get_rnd_timestep(num_samples, dist="normal"):
  "Sample noise level from given distribution"
  if dist == "normal":
    sigmas = torch.randn((num_samples,)).sigmoid()
  elif dist == "uniform":
    sigmas = torch.rand((num_samples,))
  elif dist in ["beta", "beta-high"]:
    if dist == "beta": 
      alpha, beta = 1, 2.5
    else: 
      alpha, beta = 2.5, 1        
    beta_dist = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(beta))
    sigmas = beta_dist.sample([num_samples])
  else:
      raise Exception(f"unknown distribution {dist}")
  return sigmas

def add_noise(latent, noise, timestep):
  "Add given noise at given level (`timestep`) to latent"
  return (1 - timestep) * latent + timestep * noise # (1-noise_level) * latent + noise_level * noise   

def get_schedule(num_steps, rho=5):
  "Karras et al schedule for sigma_max = 1 and sigma_min = 0"
  return torch.linspace(1, 0, num_steps + 1) ** (1/rho)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--dummy", action="store_true")
  args = parser.parse_args()
  train(mock=args.dummy)
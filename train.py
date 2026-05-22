import os
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from itertools import cycle
from einops import rearrange
from PIL import Image
from datetime import date
from random_slugs import generate_slug
from pathlib import Path

from flux2klein import (
  load_transformer_flux2klein4base,
  load_ae,
  ae_encode,
  ae_decode,
  prc_txt, 
  prc_img, 
  Flux2KleinInputs
)
from core.images import pil_cat, pil_add_text, match_width_keep_aspect

def train(
  device = "cuda" if torch.cuda.is_available() else "mps",
  dtype = torch.bfloat16,
  dataset = "g-ronimo/masked_background_v7",
  lr = 1e-5,
  steps = 2001,
  seed = 42,
  steps_log = 1,
  steps_eval = 50,
  mock = True,
):
  """
    Trains flux2klein 4b base as an object remover. 
    No guidance yet
    WIP! 
  """
  if seed is not None:
    torch.manual_seed(seed)

  run_name = f"{date.today().isoformat()}-{generate_slug(num_of_words=2)}"
  run_dir = Path(f"run/{run_name}")
  run_dir.mkdir()
  eval_dir = f"{run_dir}/eval"
  print(f"Run: {run_name}")

  prompt_tok = torch.load("cache/prompt_remove.pt", map_location="cpu").to(device)
  prompt_empty_tok = torch.load("cache/prompt_empty.pt", map_location="cpu").to(device)

  transformer = load_transformer_flux2klein4base(mock=mock).to(dtype).to(device)
  ae = load_ae(mock=mock).to(dtype).to(device)
  ds = load_dataset(dataset)
  data_sampler = cycle(torch.utils.data.RandomSampler(ds["train"], generator=torch.manual_seed(seed)))
  eval_images = [
    img_input
    for sample in ds["eval"]
    for img_input, img_target in [preprocess_sample(sample)]
  ]
  optimizer = (
    bnb.optim.AdamW8bit(transformer.parameters(), lr=lr) if device == "cuda" else
    torch.optim.AdamW(transformer.parameters(), lr=lr) 
  )

  for step in range(steps):
    transformer.train()

    # Sample input (=with masked area) and target image
    img_in, img_target = preprocess_sample(ds["train"][next(data_sampler)])
    if step == 0:
      log_first_sample(img_in, img_target, run_dir)
    img_in = ae_encode(ae, img_in).squeeze()
    img_target = ae_encode(ae, img_target).squeeze()

    # Gaussian noise
    noise = torch.randn_like(img_in, dtype=dtype, device=device)

    # Choose noise level
    timestep = get_rnd_timestep(1, dist="uniform").to(device).to(dtype)

    transformer_inputs = Flux2KleinInputs(
      noise = noise,
      prompt = prompt_tok,
      timestep = timestep,
      images = [ img_in ],
      img_clean = img_target
    )

    pred = transformer.forward(
      **transformer_inputs.as_dict(),
      timesteps = timestep,
      guidance = None
    )
    pred, _ = pred.split(pred.shape[1] // 2, dim=1) 

    loss = F.mse_loss(pred, transformer_inputs.get_noise() - transformer_inputs.get_target())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.detach().cpu().item()

    if step % steps_log == 0:
      print(f"Step {step} loss: {loss:.2f}, grad_norm: {grad_norm:.2f} (noise level: {timestep.item():.2f})")

    if step % steps_eval == 0:
      eval_step(step, transformer, ae, prompt_tok, prompt_empty_tok, eval_images, eval_dir)

def log_first_sample(img_in, img_target, run_dir):
  print("First sample image saved. Size: ", img_in.size, f"(mode {img_in.mode})")
  pil_cat(img_in, img_target).save(f"{run_dir}/sample_zero.jpg")

def eval_step(step, transformer, ae, prompt_tok, prompt_empty_tok, images, eval_dir):
  transformer.eval()

  eval_dir = Path(eval_dir)
  eval_dir.mkdir(exist_ok=True)
  images_dir = eval_dir / "images"
  images_dir.mkdir(exist_ok=True)

  gallery = None
  for i, image in enumerate(images):
    image_gen_cfg0 = pil_add_text(
      pil_cat(
        image, img2img(transformer, ae, prompt_tok, image, guidance = None, num_steps=50)
      ), 
      "CFG None"
    )
    image_gen_cfg4 = pil_add_text(
      pil_cat(
        image, img2img(transformer, ae, prompt_tok, image, guidance = 4, prompt_neg_tok = prompt_empty_tok, num_steps=50)
      ), 
      "CFG 4"
    )
    image_out = pil_cat(image_gen_cfg0, image_gen_cfg4, hor=False)
    image_out.save(images_dir / f"eval-{step}_output-{i}.jpg")
    if gallery is None:
      gallery = image_out
    else:
      gallery = pil_cat(gallery, match_width_keep_aspect(image_out, gallery), hor=False)

  gallery_fn = eval_dir / f"eval-{step}_gallery.jpg"
  gallery.save(gallery_fn)
  print(f"Eval at step {step}. Eval gallery written to {gallery_fn.name}")

  torch.cuda.empty_cache()

def img2img(
  transformer,
  ae,
  prompt_tok,
  img_ref,
  prompt_neg_tok = None,
  num_steps = 50,
  seed = 42,
  guidance = None,
  device = "cuda" if torch.cuda.is_available() else "mps",
  dtype = torch.bfloat16,
  output_fn = "output.jpg"
  ):
  if seed is not None:
    torch.manual_seed(seed)
  assert (guidance is None and prompt_neg_tok is None) or (guidance is not None and prompt_neg_tok is not None)

  img_w, img_h = img_ref.size
  noise = torch.randn([128, img_h//16, img_w//16], dtype=dtype, device=device,
   # generator=torch.manual_seed(seed)
  )
  img_ref = ae_encode(ae, img_ref).squeeze()

  transformer_inputs = Flux2KleinInputs(
    noise = noise,
    prompt = prompt_tok,
    prompt_neg = prompt_neg_tok,
    images = [ img_ref ],
  )

  timesteps = get_schedule(num_steps)

  for step, (t_curr, t_next) in enumerate(
    tqdm(zip(timesteps, timesteps[1:]), desc="Denoising", total=num_steps)
    ):
    timesteps_vec = torch.full((1,), t_curr, device=device, dtype=dtype)    

    with torch.no_grad():
      pred = transformer.forward(
        **transformer_inputs.as_dict(),
        timesteps = timesteps_vec,
        guidance = None
      )
      # model returns [img + img_refs] -> strip img_refs
      img = transformer_inputs.get_img_noisy()
      pred = pred[:, :img.shape[1]]

      if guidance:
        pred_cond, pred_uncond = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)

    img = transformer_inputs.get_img_noisy()
    img = img + (t_next - t_curr) * pred
    transformer_inputs.update_img_noisy(img)

  # unflatten tensor; flat -> 2d  
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

def get_schedule(num_steps, rho=5):
  "Karras et al schedule for sigma_max = 1 and sigma_min = 0"
  return torch.linspace(1, 0, num_steps + 1) ** (1/rho)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--dummy", action="store_true")
  args = parser.parse_args()
  train(mock=args.dummy)
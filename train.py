import torch
import torch.nn.functional as F
import huggingface_hub
import bitsandbytes as bnb
from safetensors.torch import load_file as load_sft
from torch.utils.data import RandomSampler
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from contextlib import contextmanager
from time import perf_counter
from itertools import cycle
from PIL import Image
from pathlib import Path
from einops import rearrange

from flux2_src.model import Flux2, Klein4BParams
from flux2_src.autoencoder import AutoEncoder, AutoEncoderParams
from flux2_src.sampling import prc_txt, prc_img

def train(
  device = "cuda",
  dtype = torch.bfloat16,
  dataset = "g-ronimo/masked_background_v6",
  lr = 1e-4,
  steps = 4001,
  seed = 42,
):
  """
    Trains flux2klein 4b base as an object remove. 
    No guidance yet
    WIP! 
  """
  prompt_tok = torch.load("data/prompt_remove.pt")
  transformer = load_transformer_flux2klein4base().to(dtype).to(device)
  ae = load_ae().to(dtype).to(device)
  ds = load_dataset(dataset)["train"]
  data_sampler = cycle(torch.utils.data.RandomSampler(ds, generator=torch.manual_seed(seed)))
  optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=lr)
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
  device = "cuda",
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

def load_transformer_mock():
  "Load rnd. weight tiny FLUX2"
  with catchtime() as time_taken:
    transformer = Flux2(
      Klein4BParams(
        depth=1, 
        depth_single_blocks=1,
        # hidden_size=3048
      )
    )
  print(f"Transformer loaded in {time_taken():.1f}s")

  return transformer 

def load_transformer_flux2klein4base():
  "Load rnd. weight tiny FLUX2"
  model_params = Klein4BParams()
  with catchtime() as time_taken:
    with torch.device("meta"):
      transformer = Flux2(model_params)

    weight_path = huggingface_hub.hf_hub_download(
      # repo_id="black-forest-labs/FLUX.2-klein-4B",
      repo_id="black-forest-labs/FLUX.2-klein-base-4B",
      filename='flux-2-klein-base-4b.safetensors',
      # filename='flux-2-klein-4b.safetensors',
      repo_type="model",
    )
    sd = load_sft(weight_path)
    transformer.load_state_dict(sd, strict=True, assign=True)
  print(f"Flow model loaded in {time_taken():.1f}s")

  return transformer 



def load_ae():
  "Load FLUX2 AE"
  with catchtime() as time_taken:
    ae = AutoEncoder(AutoEncoderParams())

    weight_path = huggingface_hub.hf_hub_download(
      repo_id="black-forest-labs/FLUX.2-dev",
      filename="ae.safetensors",
      repo_type="model",
    )

    sd = load_sft(weight_path, device="cpu")
    ae.load_state_dict(sd, strict=True, assign=True)
    ae = ae.eval()
  print(f"AutoEncoder loaded in {time_taken():.1f}s")

  return ae

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
    if dist == "beta": alpha, beta = 1, 2.5
    else: alpha, beta = 2.5, 1        
    beta_dist = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(beta))
    sigmas = beta_dist.sample([num_samples])
  else:
      raise Exception(f"unknown distribution {dist}")
  return sigmas

def add_noise(latent, noise, timestep):
  "Add given noise at given level (`timestep`) to latent"
  return (1 - timestep) * latent + timestep * noise # (1-noise_level) * latent + noise_level * noise   

def ae_decode(ae, img_latent):
  "Latent (Tensor) -> Image (PIL)"
  _, _, h, w = img_latent.shape

  device = next(ae.parameters()).device

  with torch.no_grad():
    img = ae.decode(img_latent.to(device)).detach()

  img.squeeze_()

  # first clamp, then normalize - artifacts if the other way around
  img = img.clamp(-1, 1)
  img = img * 0.5 + 0.5
  # num,py doesnt like bfloat16
  img = transforms.ToPILImage()(img.to(torch.float32))

  return img

def ae_encode(ae, img):
  "Image (PIL) -> Latent (Tensor)"

  preprocess = transforms.Compose([
    # height and width have to be divisible by 16 -> crop from center
    transforms.CenterCrop(tuple(x//16*16 for x in (img.height, img.width))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    lambda x: x.to(device).unsqueeze(0)  # add batch dim
  ])

  # Get AE device and dtype from the first parameter
  device = next(ae.parameters()).device
  dtype = next(ae.parameters()).dtype

  img = preprocess(img).to(device).to(dtype)

  with torch.no_grad():
    img_latent = ae.encode(img)

  return img_latent

def pil_cat(img1, img2, hor = True):
  "Concat two PIL Images"
  if hor is None:
    hor = True if max([i.width/i.height for i in (img1, img2)]) < 1 else False
  img = (
    Image.new("RGB", (img1.width+img2.width, max(img1.height, img2.height))) if hor else
    Image.new("RGB", (max(img1.width, img2.width), img1.height+img2.height))
  )
  img.paste(img1, (0, 0))
  img.paste(img2, (img1.width, 0) if hor else (0, img1.height))
  return img

def get_schedule(num_steps, rho=5):
  "Karras et al schedule for sigma_max = 1 and sigma_min = 0"
  return torch.linspace(1, 0, num_steps + 1) ** (1/rho)

@contextmanager
def catchtime():
  t1 = t2 = perf_counter() 
  yield lambda: t2 - t1
  t2 = perf_counter() 


if __name__ == "__main__":
  train()
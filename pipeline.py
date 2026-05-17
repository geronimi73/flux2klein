import torch
import huggingface_hub
import plotext as plt
from pathlib import Path
from safetensors.torch import load_file as load_sft
from torchvision import transforms
from contextlib import contextmanager
from time import perf_counter
from einops import rearrange
from tqdm import tqdm
from PIL import Image

# BFL model defs
from flux2_src.text_encoder import Qwen3Embedder
from flux2_src.autoencoder import AutoEncoder, AutoEncoderParams
from flux2_src.model import Flux2, Klein4BParams

device = "cuda"
dtype = torch.bfloat16
text_encoder, ae, flow_model = [None] * 3

@contextmanager
def catchtime():
  t1 = t2 = perf_counter() 
  yield lambda: t2 - t1
  t2 = perf_counter() 

# source: flux2.sampling
def prc_txt(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  _l, _ = x.shape  # noqa: F841

  coords = {
    "t": torch.arange(1) if t_coord is None else t_coord,
    "h": torch.arange(1),  # dummy dimension
    "w": torch.arange(1),  # dummy dimension
    "l": torch.arange(_l),
  }
  x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
  return x, x_ids.to(x.device)

# source: flux2.sampling
def prc_img(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  _, h, w = x.shape  # noqa: F841
  x_coords = {
      "t": torch.arange(1) if t_coord is None else t_coord,
      "h": torch.arange(h),
      "w": torch.arange(w),
      "l": torch.arange(1),
  }
  x = rearrange(x, "c h w -> (h w) c")
  x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
  return x, x_ids.to(x.device)

def plot_schedule(timesteps, title="Timestep schedule"):
  "Print ascii plot of timestep schedule"
  plt.canvas_color("default")
  plt.axes_color("default")
  plt.ylabel("%Noise")
  plt.plot(timesteps)
  plt.title("Timestep schedule")
  plt.plotsize(50, 20)  
  plt.show()

def load_flow_model():
  global flow_model
  if flow_model is not None: return
  model_params = Klein4BParams()
  with catchtime() as time_taken:
    with torch.device("meta"):
      flow_model = Flux2(model_params).to(torch.bfloat16)

    weight_path = huggingface_hub.hf_hub_download(
      repo_id="black-forest-labs/FLUX.2-klein-4B",
      filename='flux-2-klein-4b.safetensors',
      repo_type="model",
    )
    sd = load_sft(weight_path, device=str(device))
    flow_model.load_state_dict(sd, strict=True, assign=True)
  print(f"Flow model loaded in {time_taken():.1f}s")

def load_ae():
  global ae
  if ae is not None: return
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

def ae_decode(img_latent):
  "Latent (Tensor) -> Image (PIL)"
  _, _, h, w = img_latent.shape

  ae.to(device)

  with torch.no_grad():
    img = ae.decode(img_latent.to(device)).cpu()

  img.squeeze_()

  # first clamp, then normalize - artifacts if the other way around
  img = img.clamp(-1, 1)
  img = img * 0.5 + 0.5
  img = transforms.ToPILImage()(img)

  ae.to("cpu")
  torch.cuda.empty_cache()

  return img

def ae_encode(img):
  "Image (PIL) -> Latent (Tensor)"

  preprocess = transforms.Compose([
    # height and width have to be divisible by 16 -> crop from center
    transforms.CenterCrop(tuple(x//16*16 for x in (img.height, img.width))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    lambda x: x.to(device).unsqueeze(0)  # add batch dim
  ])
  img = preprocess(img)

  ae.to(device)

  with torch.no_grad():
    img_latent = ae.encode(img).cpu()

  ae.to("cpu")
  torch.cuda.empty_cache()

  return img_latent

def encode_prompt(prompt):
  "Load Text Encoder and encode text prompt"
  global text_encoder
  if text_encoder is None:
    with catchtime() as time_taken:
      text_encoder = Qwen3Embedder(model_spec=f"Qwen/Qwen3-4B", device="cpu").to(dtype)
    print(f"Text encoder loaded in {time_taken():.1f}s")
  text_encoder.to(device)
  emb = text_encoder([prompt]).squeeze()
  text_encoder.to("cpu")
  torch.cuda.empty_cache()

  return emb

def get_schedule(num_steps, rho=5):
  "Karras et al schedule for sigma_max = 1 and sigma_min = 0"
  return torch.linspace(1, 0, num_steps + 1) ** (1/rho)

def text2img(
  prompt,
  img_h = 1024,
  img_w = 1024,
  num_steps = 4,
  output_prefix = "output/generated_image",
  seed = 42
  ):
  if seed is not None:
    torch.manual_seed(seed)
    
  txt = encode_prompt(prompt).to(dtype)
  txt, txt_ids = prc_txt(txt)
  img = torch.randn([128, img_h//16, img_w//16], dtype=dtype, device=device)
  img, img_ids = prc_img(img)

  # Add batch dim. 
  txt, txt_ids = txt[None,], txt_ids[None,]
  img, img_ids = img[None,], img_ids[None,]

  load_ae()
  load_flow_model()
  timesteps = get_schedule(num_steps)
  plot_schedule(timesteps)

  guidance_vec = torch.full((img.shape[0],), 1.0, device=device, dtype=dtype)

  for step, (t_curr, t_next) in enumerate(
    tqdm(zip(timesteps, timesteps[1:]), desc="Denoising", total=num_steps)
    ):
    timesteps_vec = torch.full((img.shape[0],), t_curr, device=device, dtype=dtype)    

    with torch.no_grad():
      pred = flow_model.forward(
        x = img, x_ids = img_ids,
        ctx = txt, ctx_ids = txt_ids,
        timesteps = timesteps_vec,
        guidance = guidance_vec
      )
    img = img + (t_next - t_curr) * pred

    # decode and save intermed. results
    output_fn = (
      f"{output_prefix}-{step}-{t_next:.2f}.jpg" if t_next > 0 
      else output_prefix + ".jpg"
    )

    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    # unflatten tensor; linear -> 2d
    ae_decode(
      rearrange(img, "1 (h w) c -> 1 c h w", h=img_h//16)
    ).save(output_fn)
  print("Output saved to", output_fn)

def img2img(
  prompt,
  img_refs,
  img_h = 1024,
  img_w = 1024,
  num_steps = 4,
  output_prefix = "output/modified_image",
  seed = 42
  ):
  # 99% same as text2img(), changes as comments
  if seed is not None:
    torch.manual_seed(seed)
  
  txt = encode_prompt(prompt).to(dtype)
  txt, txt_ids = prc_txt(txt)
  img = torch.randn([128, img_h//16, img_w//16], dtype=dtype, device=device)
  img, img_ids = prc_img(img)

  # ref. images -> latents, separate `t` dim for each img.
  load_ae()
  img_refs = [
    ae_encode(img_ref).to(device).to(torch.bfloat16)
    for img_ref in img_refs
  ]
  # t_coord: time offsets for each image. noise: t=0, ref1: t=10, , ref2: t=20, etc.
  img_refs, img_refs_ids = zip(*[
    prc_img(img_ref.squeeze(), t_coord=torch.tensor([(idx+1)*10], dtype=torch.int64))
    for idx, img_ref in enumerate(img_refs)
  ])

  # Add batch dim. 
  txt, txt_ids = txt[None,], txt_ids[None,]
  img, img_ids = img[None,], img_ids[None,]
  # img_refs is a list of tensors
  img_refs = [i[None,] for i in img_refs] 
  img_refs_ids = [i[None,] for i in img_refs_ids] 

  load_flow_model()
  timesteps = get_schedule(num_steps)
  plot_schedule(timesteps)

  guidance_vec = torch.full((img.shape[0],), 1.0, device=device, dtype=dtype)

  for step, (t_curr, t_next) in enumerate(
    tqdm(zip(timesteps, timesteps[1:]), desc="Denoising", total=num_steps)
    ):
    timesteps_vec = torch.full((img.shape[0],), t_curr, device=device, dtype=dtype)    

    with torch.no_grad():
      pred = flow_model.forward(
        # x: Concatenated img(=noise)+ref imgs; "id" coords encoded with different `t` dim for each img. that's the most important point
        x = torch.cat([img] + img_refs, dim=1), 
        x_ids = torch.cat([img_ids] + img_refs_ids, dim=1),
        ctx = txt, ctx_ids = txt_ids,
        timesteps = timesteps_vec,
        guidance = guidance_vec
      )
      # model returns [img + img_refs] -> strip img_refs
      pred = pred[:, :img.size(1)]
    img = img + (t_next - t_curr) * pred

    # decode and save intermed. results
    output_fn = (
      f"{output_prefix}-{step}-{t_next:.2f}.jpg" if t_next > 0 
      else output_prefix + ".jpg"
    )

  Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
  # unflatten tensor; linear -> 2d  
  ae_decode(
    rearrange(img, "1 (h w) c -> 1 c h w", h=img_h//16)
  ).save(output_fn)
  print("Output saved to", output_fn)


if __name__ == "__main__":
  # Usage examples
  # text2img("A beautiful mountain landscape", output_prefix = "output/generated_mountain")
  
  img2img(
    "Remove and replace the tanks in image 1. Replace them with the flowers of image 2.",
    img_refs = [
      img_ref := Image.open("assets/taiwan.jpg"),
      Image.open("assets/trump_and_flowers.jpg"),
    ],
    img_h = img_ref.height,
    img_w = img_ref.width,
    output_prefix = "output/edit_tank_flowers",
  )



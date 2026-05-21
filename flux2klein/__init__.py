import torch
import huggingface_hub
from safetensors.torch import load_file as load_sft
from torchvision import transforms

from .flux2_src.model import Flux2, Klein4BParams
from .flux2_src.autoencoder import AutoEncoder, AutoEncoderParams
from .flux2_src.sampling import prc_txt, prc_img
from core.utils import catchtime
from core.latents import add_noise

__all__ = ["prc_txt", "prc_img"]

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

def load_transformer_flux2klein4base(mock=False):
  "Load rnd. weight tiny FLUX2"
  with catchtime() as time_taken:
    if mock:
      transformer = Flux2(Klein4BParams(depth=1, depth_single_blocks=1))
    else: 
      with torch.device("meta"):
        transformer = Flux2(Klein4BParams())

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

def load_ae(mock=False):
  "Load FLUX2 AE"
  with catchtime() as time_taken:
    ae = AutoEncoder(AutoEncoderParams())

    if not mock:
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
  # numpy doesnt like bfloat16
  img = transforms.ToPILImage()(img.to(torch.float32))

  return img

def ae_encode(ae, img, patch_size=16):
  "Image (PIL) -> Latent (Tensor)"

  preprocess = transforms.Compose([
    # height and width have to be divisible by 16 -> crop from center
    transforms.CenterCrop(tuple(x//patch_size*patch_size for x in (img.height, img.width))),
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

class Flux2KleinInputs:
  def __init__(self, 
    noise,
    prompt,
    timestep = None,
    images = None,
    img_clean = None    # for training
  ):
    # need: x, x_ids, ctx, ctx_ids
    # x = images = noise + ref images
    # ctx = prompt tokens

    assert len(noise.shape) == 3

    if img_clean is not None:
      # training
      assert len(img_clean.shape) == 3
      assert timestep is not None
      img_noisy = add_noise(img_clean, noise, timestep)
    else:
      # inference
      assert timestep is None
      img_noisy = noise

    # Build x and x_ids; img=noisy latent, img_refs=reference images
    img, img_ids = prc_img(img_noisy)
    img, img_ids = img[None,], img_ids[None,] # add batch dim.

    if images:
      img_refs, img_refs_ids = zip(*[
        prc_img(img_ref.squeeze(), t_coord=torch.tensor([(idx+1)*10], dtype=torch.int64))
        for idx, img_ref in enumerate(images)
      ])
      img_refs = torch.cat(img_refs, dim=0)[None, ]
      img_refs_ids = torch.cat(img_refs_ids, dim=0)[None, ]

    self.x =     torch.cat([img, img_refs], dim=1) if images else img
    self.x_ids = torch.cat([img_ids, img_refs_ids], dim=1) if images else img_ids
    self.ctx, self.ctx_ids = [t[None, ] for t in prc_txt(prompt)]

    # flat, with B dim.
    self.img_noisy = img
    self.img_clean = prc_img(img_clean)[0][None, ] if img_clean is not None else None
    self.noise = prc_img(noise)[0][None, ]

  def get_target(self):
    return self.img_clean

  def get_noise(self):
    return self.noise

  def as_dict(self):
    return dict(
      x = self.x,
      x_ids = self.x_ids,
      ctx = self.ctx,
      ctx_ids = self.ctx_ids,
    )

  def get_img_noisy(self):
    return self.img_noisy

  def update_img_noisy(self, img_noisy):
    # B T C
    self.img_noisy = img_noisy
    self.x[:, :img_noisy.shape[1], :] = img_noisy

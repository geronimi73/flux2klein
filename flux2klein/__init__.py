import torch
import huggingface_hub
from safetensors.torch import load_file as load_sft
from torchvision import transforms

from .flux2_src.model import Flux2, Klein4BParams
from .flux2_src.autoencoder import AutoEncoder, AutoEncoderParams
from .flux2_src.sampling import prc_txt, prc_img
from core.utils import catchtime

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


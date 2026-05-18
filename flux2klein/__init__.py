import torch
import huggingface_hub
from safetensors.torch import load_file as load_sft

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


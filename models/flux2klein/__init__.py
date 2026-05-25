import torch
import huggingface_hub
from torch import Tensor
from safetensors.torch import load_file as load_sft
from torchvision import transforms
from typing import List

from core.utils import catchtime
from .flux2_src.model import Flux2, Klein4BParams
from .flux2_src.autoencoder import AutoEncoder, AutoEncoderParams
from .flux2_src.sampling import prc_txt, prc_img

__all__ = ["prc_txt", "prc_img"]

def load_transformer_mock():
  "Load rnd. weight tiny FLUX2"
  with catchtime() as time_taken:
    transformer = Flux2(
      Klein4BParams(
        depth=1, 
        depth_single_blocks=1,
        hidden_size = 256,
        num_heads = 2,
      )
    )
  print(f"Transformer loaded in {time_taken():.1f}s")

  return transformer 

def load_transformer_flux2klein4base(mock=False):
  "Load rnd. weight tiny FLUX2"
  with catchtime() as time_taken:
    if mock:
      transformer = load_transformer_mock()
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
    images_noisy: List[Tensor],           # noisy latents. list of [128, h , w]
    prompts: List[Tensor],                # encoded prompt, [512, 7680]
    images_clean: List[Tensor] = [],      # clean latents, each [128, h , w]
    ref_images: List[List[Tensor]] = None, # might be mult. ref. images per input -> list of lists of Tensors
    noise: List[Tensor] = None
  ):
    # i've fucked this up too many times
    if ref_images is None:
        ref_images = [[] for _ in range(len(images_noisy))]
    else:
        assert len(images_noisy) == len(ref_images), f"{len(images_noisy)}, {len(ref_images)}"    
        assert isinstance(ref_images[0], List)
        assert isinstance(ref_images[0][0], torch.Tensor)
    assert len(images_noisy) > 0 and len(images_noisy) == len(prompts)
    assert isinstance(images_noisy[0], torch.Tensor)
    assert len(images_noisy[0].shape) == 3
    assert all([isinstance(x, List) or x is None for x in [images_noisy, prompts, images_clean, ref_images, noise]])
    if images_clean:
      for _noise, _target in zip(images_noisy, images_clean):
        assert _noise.shape == _target.shape, f"noise {_noise.shape} != target {_target.shape}"

    # store number of tokens for noisy image
    # assume! this is the same for all images in batch
    self.input_img_tokens = images_noisy[0].shape[1] * images_noisy[0].shape[2] 
    self.num_samples = len(images_noisy)
      
    # First build x and x_ids; concatenate image and image references
    img, img_ids = [], []
    for _noise, _refs in zip(images_noisy, ref_images):
        # for each input image: process noise + ref. images
        _img, _img_ids = zip(*[    
            prc_img(img, t_coord=torch.tensor([idx*10], dtype=torch.int64))   # prc_img: [C H W] -> [T C]
            # _noise is a single Tensor [C H W], _refs is a list of Tensors
            for idx, img in enumerate([_noise] + _refs)
        ])
        # Concatenate flat image patches and ids
        img.append(torch.cat(_img))   
        img_ids.append(torch.cat(_img_ids))
    # list([T C]) for each image -> [B T C]
    self.x =     torch.stack(img, dim=0)
    self.x_ids = torch.stack(img_ids, dim=0)

    # SECOND: Build ctx = text
    prompt, prompt_ids = zip(*[
        prc_txt(prompt.squeeze())
        for idx, prompt in enumerate(prompts)
    ])
    self.ctx =     torch.stack(prompt, dim=0)
    self.ctx_ids = torch.stack(prompt_ids, dim=0)

    # For training
    self.noise =  torch.stack([prc_img(img.squeeze())[0] for img in noise]) if noise else None
    self.target = torch.stack([prc_img(img.squeeze())[0] for img in images_clean]) if images_clean else None

  def get_target(self):
    return self.target

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
    return self.x[:, :self.input_img_tokens, :] 
      
  def update_img_noisy(self, img_noisy):      
    self.x[:, :self.input_img_tokens, :] = img_noisy

import torch
import huggingface_hub
import platform

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from safetensors.torch import load_file as load_sft

device = "cuda"

def pil_add_text(image, text, position=None, font_size=None, font_color=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0)):
  font = ImageFont.truetype(
    font = "Times.ttc" if platform.system() == "Darwin" else "DejaVuSans.ttf",
    size = image.height//10 if font_size is None else font_size
  )

  img_copy = image.copy()
  ImageDraw.Draw(img_copy).text(
    xy = (image.width//20, image.height//20) if position is None else position,
    text = text, font = font, fill = font_color, stroke_width = stroke_width, stroke_fill = stroke_fill
  )
  
  return img_copy

# concat two images
def pil_cat(img1, img2, hor = None):
  if hor is None:
    hor = True if max([i.width/i.height for i in (img1, img2)]) < 1 else False
  img = (
    Image.new("RGB", (img1.width+img2.width, max(img1.height, img2.height))) if hor else
    Image.new("RGB", (max(img1.width, img2.width), img1.height+img2.height))
  )
  img.paste(img1, (0, 0))
  img.paste(img2, (img1.width, 0) if hor else (0, img1.height))
  return img

# stack input+output + blow up of center crop
def pil_compare_blowup(
  img1, img2, 
  img1_label = "input",
  img2_label = "output",
  crop_fraction = 1/5, 
  blowup = 4, 
  ):
  crop_size = int(min(img1.width, img1.height) * crop_fraction)
  cc_size = (x:=min(img1.width, img1.height), x)
  img1_cc = transforms.CenterCrop(crop_size)(img1).resize(cc_size)
  img2_cc = transforms.CenterCrop(crop_size)(img2).resize(cc_size)

  return pil_cat(
    pil_cat(pil_add_text(img1, img1_label), img1_cc, hor=True),
    pil_cat(pil_add_text(img2, img2_label), img2_cc, hor=True),
  )


def load_ae(version):
  if version == "flux1":
    from flux_src.modules.autoencoder import AutoEncoder, AutoEncoderParams

    weight_path = huggingface_hub.hf_hub_download(
      repo_id="black-forest-labs/FLUX.1-dev",
      filename="ae.safetensors",
      repo_type="model",
    )
    ae_params=AutoEncoderParams(
      resolution=256,
      in_channels=3,
      ch=128,
      out_ch=3,
      ch_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      z_channels=16,
      scale_factor=0.3611,
      shift_factor=0.1159,
    )

  else:
    from flux2_src.autoencoder import AutoEncoder, AutoEncoderParams

    weight_path = huggingface_hub.hf_hub_download(
      repo_id="black-forest-labs/FLUX.2-dev",
      filename="ae.safetensors",
      repo_type="model",
    )
    ae_params = AutoEncoderParams()

  print(f"Loading {weight_path} for the AutoEncoder weights")
  ae = AutoEncoder(ae_params)
  sd = load_sft(weight_path, device=str(device))
  ae.load_state_dict(sd, strict=True, assign=True)
  ae.eval()

  return ae

def autoencode(img_orig, version):
  assert version in ["flux1", "flux2"]

  preprocess = transforms.Compose([
    # height and width have to be divisible by 16 -> crop from center
    transforms.CenterCrop(tuple(x//16*16 for x in (img_orig.height, img_orig.width))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    lambda x: x.to(device).unsqueeze(0)  # add batch dim
  ])
  img = preprocess(img_orig)

  ae = load_ae(version)
  ae.eval()
  ae.requires_grad_ = False

  with torch.no_grad():
    latent = ae.encode(img)

  with torch.no_grad():
    img_back = ae.decode(latent).squeeze().cpu()

  # first clamp, then normalize - artifacts if the other way around
  img_back = img_back.clamp(-1, 1)
  img_back = img_back * 0.5 + 0.5
  img_back = transforms.ToPILImage()(img_back)

  return img_back

if __name__ == "__main__":
  input_fn = "assets/city_small.jpg"
  output_path = Path("output") / Path(input_fn).name

  img_orig = Image.open(input_fn)
  img_back_flux1 = autoencode(img_orig, "flux1")
  img_back_flux2 = autoencode(img_orig, "flux2")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  pil_compare_blowup(
      img_back_flux1, 
      img_back_flux2,
      img1_label = "flux1 AE",
      img2_label = "flux2 AE",
  ).save(output_path)

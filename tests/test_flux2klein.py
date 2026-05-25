import torch

from models.flux2klein import (
  load_transformer_flux2klein4base,
  load_ae,
  generate_txt2img,
)

def load_img_prompt():
  return torch.load("cache/prompt_loneroad.pt", map_location="cpu")

def load_neg_prompt():
  return torch.load("cache/prompt_empty.pt", map_location="cpu")

def get_rnd_prompt():
  return torch.randn([512, 7680])

def txt2img(mock=True, num_images=5, height=512, width=512, seed=42, num_steps=5, guidance=None):
  device="cuda"
  dtype = torch.bfloat16

  torch.manual_seed(seed)
  if mock:
    prompt_tok = get_rnd_prompt().to(device).to(dtype)
    prompt_empty_tok = get_rnd_prompt().to(device).to(dtype)
  else:
    prompt_tok = load_img_prompt().to(device).to(dtype)
    prompt_empty_tok = load_neg_prompt().to(device).to(dtype)

  transformer = load_transformer_flux2klein4base(mock=mock).to(dtype).to(device)
  ae = load_ae(mock=mock).to(dtype).to(device)

  return generate_txt2img(
    [prompt_tok] * num_images,
    transformer,
    ae,
    seed = seed,
    height = height,
    width = width,
    num_steps = num_steps,
    guidance = guidance,
    prompt_neg = prompt_empty_tok if guidance is not None else None
  )
  
def test_txt2img():
  if not torch.cuda.is_available():
    return

  txt2img(mock=True, num_images=1, height=64, width=64, num_steps=5, guidance=None)
  txt2img(mock=True, num_images=2, height=64, width=64, num_steps=5, guidance=None)
  txt2img(mock=True, num_images=1, height=64, width=64, num_steps=5, guidance=4.0)
  txt2img(mock=True, num_images=2, height=64, width=64, num_steps=5, guidance=4.0)

if __name__ == "__main__":
  images = txt2img(mock=False, num_images=2, height=512, width=512, num_steps=50, guidance=None)
  for i, img in enumerate(images):
    img.save(f"test-{i}_CFG0.jpg")

  images = txt2img(mock=False, num_images=2, height=512, width=512, num_steps=50, guidance=4.0)
  for i, img in enumerate(images):
    img.save(f"test-{i}_CFG4.jpg")

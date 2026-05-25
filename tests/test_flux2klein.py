import torch
from tqdm import tqdm
from einops import rearrange

from models.flux2klein import (
  Flux2KleinInputs,
  load_transformer_flux2klein4base,
  load_ae,
  ae_decode,
)
from core.sampling import get_schedule

def get_rnd_prompt():
    return torch.randn([512, 7680])

def txt2img_mock(img_h=512, img_w=512, bs=5, seed=42, num_steps=5, guidance=None):
  if not torch.cuda.is_available():
    return

  device="cuda"
  dtype = torch.bfloat16

  torch.manual_seed(seed)
  prompt_tok = get_rnd_prompt().to(device).to(dtype)
  prompt_empty_tok = get_rnd_prompt().to(device).to(dtype)

  transformer = load_transformer_flux2klein4base(mock=True).to(dtype).to(device)
  ae = load_ae(mock=True).to(dtype).to(device)

  prompts = [prompt_tok] * bs
  noise = [
    torch.randn([128, img_h//16, img_w//16], dtype=dtype, device=device)
    for _ in range(bs)
  ]

  if guidance:
    noise   += noise 
    prompts += [prompt_empty_tok] * bs

  transformer_inputs = Flux2KleinInputs(
    images_noisy =  noise,
    prompts =       prompts,
  )

  timesteps = get_schedule(num_steps)

  for step, (t_curr, t_next) in enumerate(
    tqdm(zip(timesteps, timesteps[1:]), desc="Denoising", total=num_steps)
    ):
    timesteps_vec = torch.full((transformer_inputs.num_samples,), t_curr, device=device, dtype=dtype)    

    with torch.no_grad():
      pred = transformer.forward(
        **transformer_inputs.as_dict(),
        timesteps = timesteps_vec,
        guidance = None
      )

      # batch = cond + uncond
      if guidance:
        pred_cond, pred_uncond  = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = torch.cat([pred, pred], dim=0)

    img = transformer_inputs.get_img_noisy()
    img = img + (t_next - t_curr) * pred
    transformer_inputs.update_img_noisy(img)

  # batch = [cond1, cond2, .. uncond1, uncond2, .. ] if guidance
  if guidance:
    img, _ = img.chunk(2)

  # batch = [cond1, cond2, .. ] 
  # batch of images -> list of images
  imgs = img.chunk(bs)

  # unflatten tensor; flat -> 2d  
  return [
    ae_decode(ae, rearrange(img, "1 (h w) c -> 1 c h w", h=img_h//16))
    for img in imgs
  ]
  
def test_txt2img():
  txt2img_mock(bs=5, img_h=512, img_w=512, num_steps=10, guidance=None)
  txt2img_mock(bs=5, img_h=512, img_w=512, num_steps=10, guidance=4.0)



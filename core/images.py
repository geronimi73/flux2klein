import platform
from PIL import Image, ImageDraw, ImageFont

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

def match_width_keep_aspect(source_img, target_img):
  # Get the width of the target image
  target_width = target_img.width

  # Calculate the new height to preserve the aspect ratio
  width_ratio = target_width / source_img.width
  new_height = int(source_img.height * width_ratio)

  # Resize the source image
  return source_img.resize((target_width, new_height), Image.LANCZOS)


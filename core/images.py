from PIL import Image

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

def match_width_keep_aspect(source_img, target_img):
    # Get the width of the target image
    target_width = target_img.width

    # Calculate the new height to preserve the aspect ratio
    width_ratio = target_width / source_img.width
    new_height = int(source_img.height * width_ratio)

    # Resize the source image
    return source_img.resize((target_width, new_height), Image.LANCZOS)

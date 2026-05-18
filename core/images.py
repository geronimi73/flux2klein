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

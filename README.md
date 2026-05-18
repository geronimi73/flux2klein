Teaching myself how FLUX2 klein works. 

* `autoencoder.py`: Encode/decode images, compare FLUX2 AE to FLUX1 AE
* `pipeline.py`: txt2img and img2img

Blog post: https://medium.com/@geronimo7/flux-2-klein-how-inference-works-05553fcdbe7e

## Inference

### AE

<img width="768" height="512" alt="city_small" src="https://github.com/user-attachments/assets/97c33c87-a2e3-4725-a082-2e43adff463b" />

### txt2img

https://github.com/user-attachments/assets/a9794915-9c6e-4d96-87b1-e4fc29b34848

### img2img
<img width="955" height="721" alt="taiwan" src="https://github.com/user-attachments/assets/a1803a65-0e72-4551-8f42-35b07e9a5c30" />

<img width="1280" height="720" alt="trump_and_flowers" src="https://github.com/user-attachments/assets/d365ed94-90fa-4a73-9bda-ea829d7cc7d3" />

<img width="1904" height="1440" alt="edit_tank_flowers" src="https://github.com/user-attachments/assets/1c46fa23-f353-4a3e-b652-11c7bc0d5ef8" />

## Train

* Flux2 klein 4b
* trained as object remover
* no guidance during eval
* lr 1e-4, bs 1, result after 400 steps

<img width="1344" height="512" alt="eval-400_output" src="https://github.com/user-attachments/assets/7052d77d-5243-4d92-8f99-29e5310179d7" />






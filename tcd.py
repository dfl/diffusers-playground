import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

from diffusers.utils import logging
logging.set_verbosity(logging.DEBUG)

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


device = "mps"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

steps = 4
seed = 0

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."

image = pipe(
    prompt=prompt,
    num_inference_steps=steps,
    guidance_scale=0,
    eta=0.3, 
    # generator=torch.manual_seed(seed),
    generator=torch.Generator(device=device).manual_seed(0),
    # output_type="np"
).images[0]
image.save(f"otto_steps-{steps}_seed-{seed}.png")

# print(np.abs(image).sum())

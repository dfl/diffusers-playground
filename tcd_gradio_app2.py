import random

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from compel import Compel

from scheduling_tcd import TCDScheduler
from utils import save_image_with_geninfo, crc_hash, parse_params_from_image, str2num, default_torch_device
from PIL import Image

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

output_format = "jpg" #png"

css = """
h1 {
    text-align: center;
    display:block;
}
h3 {
    text-align: center;
    display:block;
}

button.tool {
    max-width: 2.2em;
    min-width: 2.2em !important;
    height: 2.4em;
    align-self: end;
    line-height: 1em;
    border-radius: 0.5em;
}
"""

device = "mps"
base_model_id = "stabilityai/stable-diffusion-2-1-base"
tcd_lora_id = "h1t/TCD-SD21-base-LoRA"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
    # torch_dtype=torch.float32,  # Changed from torch.float16
    # variant=None  # No fp16 variant needed
).to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
    # torch_dtype=torch.float32,  # Changed from torch.float16
    # variant=None  # No fp16 variant needed
).to(device)
img2img_pipe.scheduler = TCDScheduler.from_config(img2img_pipe.scheduler.config)

img2img_pipe.load_lora_weights(tcd_lora_id)
img2img_pipe.fuse_lora()

def newSeed() -> int:
    return int(random.randrange(4294967294))

def first_pass_inference(prompt, negative_prompt="", steps=4, seed=-1, eta=0.3, cfg=0, width=512, height=512, preImage=None):
    if seed is None or seed == '' or seed == -1:
        seed = newSeed()
    
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # Convert preImage to PIL Image if it's a filepath
    original_image_path = None
    if preImage is not None:
        if isinstance(preImage, str):
            original_image_path = preImage
            preImage = Image.open(preImage)
        
        # Resize pre-image to target dimensions
        preImage = preImage.resize((width, height), Image.LANCZOS)
        
        # Always use img2img pipeline when a pre-image is provided
        conditioning = compel_proc(prompt)
        neg_conditioning = compel_proc(negative_prompt)
        
        low_res_image = img2img_pipe(
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            num_inference_steps=steps,
            guidance_scale=cfg,
            eta=eta,
            generator=generator,
            image=preImage,
            strength=0.5  # Moderate strength for refinement
        ).images[0]
    else:
        # Standard text-to-image generation
        conditioning = compel_proc(prompt)
        neg_conditioning = compel_proc(negative_prompt)
        
        mheight = max(1024, height // 2)
        mwidth = max(1024, width // 2)
        
        low_res_image = pipe(
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            num_inference_steps=steps,
            guidance_scale=cfg,
            eta=eta,
            generator=generator,
            height=mheight,
            width=mwidth,
        ).images[0]
    
    return low_res_image, seed, original_image_path

def high_res_inference(low_res_image, prompt, negative_prompt, steps, seed, eta, cfg, hires_strength, width, height, original_image_path=None):
    # Convert low_res_image to PIL Image if it's a string (file path)
    if isinstance(low_res_image, str):
        low_res_image = Image.open(low_res_image)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    # Prepare conditioning
    conditioning = compel_proc(prompt)
    neg_conditioning = compel_proc(negative_prompt)
    
    # Resize for high-res
    resized_image = low_res_image.resize((height, width), Image.LANCZOS)

    # High-res refinement
    high_res_image = img2img_pipe(
        prompt_embeds=conditioning,
        negative_prompt_embeds=neg_conditioning,
        num_inference_steps=steps,
        guidance_scale=cfg,
        eta=eta,
        generator=generator,
        height=height,
        width=width,
        image=resized_image,
        strength=hires_strength
    ).images[0]
    
    # Metadata and saving logic
    metadata = {
        "seed": seed,
        "steps": steps,
        "eta": eta,
        "cfg": cfg,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoise": hires_strength,
        "model": base_model_id,
        "width": width,
        "height": height,
        "img2img": os.path.basename(original_image_path)
    }
    
    # Save high-res image
    high_res_path = f"outputs/TCD2_high_res_seed-{seed}_steps-{steps}_{crc_hash(repr(metadata))}.{output_format}"
    save_image_with_geninfo(high_res_image, str(metadata), high_res_path)
    
    return high_res_image

# Define style
title = "<h1>Trajectory Consistency Distillation (SD 2.1)</h1>"
description = "<h3>Unofficial Gradio demo for Trajectory Consistency Distillation</h3>"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/' target='_blank'>Trajectory Consistency Distillation</a> | <a href='https://github.com/jabir-zheng/TCD' target='_blank'>Github Repo</a></p>"


default_prompt = "" #Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."
examples = [
    [
        "Beautiful woman, bubblegum pink, lemon yellow, minty blue, futuristic, high-detail, epic composition, watercolor.",
        4
    ],
    [
        "Beautiful man, bubblegum pink, lemon yellow, minty blue, futuristic, high-detail, epic composition, watercolor.",
        8
    ],
    [
        "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna.",
        16
    ],
    [
        "closeup portrait of 1 Persian princess, royal clothing, makeup, jewelry, wind-blown long hair, symmetric, desert, sands, dusty and foggy, sand storm, winds bokeh, depth of field, centered.",
        16
    ],
]

def get_params_from_image(img) -> (str, str, int, int, float, float, float, int, int):
    p = parse_params_from_image(img)
    prompt = p.get('prompt', '')
    negative_prompt = p.get('negative_prompt', '')
    seed = p.get('seed', -1)
    steps = p.get('steps', 4)
    eta = p.get('eta', 0.3)
    cfg = p.get('cfg', 1.0)
    hires_strength = p.get('denoise', 0.5)  # Default to 0.5 if not found
    width = p.get('width', 512)
    height = p.get('height', 512)

    return prompt, negative_prompt, steps, seed, eta, cfg, hires_strength, width, height

with gr.Blocks(css=css) as demo:
    gr.Markdown(f'# {title}\n### {description}')
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label='Prompt', value=default_prompt)
            negative_prompt = gr.Textbox(label='Negative Prompt', value=default_prompt)
            steps = gr.Slider(
                label='Inference steps',
                minimum=4,
                maximum=16,
                value=4,
                step=1,
            )
            
            # with gr.Accordion("Advanced Options", open=True):
            with gr.Column():
                with gr.Row():
                    seed = gr.Number(label="Random Seed", value=-1)
                    luckyButton = gr.Button(value="🍀", elem_classes="tool") #tooltip="Generate new random seed", 
                    randButton = gr.Button(value="🎲", elem_classes="tool") # tooltip="Set seed to -1, which will cause a new random number to be used every time", 
                    recycleButton = gr.Button(value="♻️", elem_classes="tool") # tooltip="Reuse seed from last generation", 

            with gr.Column():
                with gr.Row():
                    eta = gr.Slider(
                            label='Gamma',
                            minimum=0.,
                            maximum=1.,
                            value=0.3,
                            step=0.1,
                        )
                    cfg = gr.Slider(
                            label='Guidance Scale (CFG)',
                            minimum=1,
                            maximum=3.,
                            value=1.,
                            step=0.05,
                        )
                with gr.Row():
                    width = gr.Slider(
                            label='Width',
                            minimum=512,
                            maximum=2048,
                            value=1536,
                            step=128,
                        )
                    height = gr.Slider(
                            label='Height',
                            minimum=512,
                            maximum=2048,
                            value=1536,
                            step=128,
                        )
                    hires_strength = gr.Slider(
                        label='Denoise Strength',
                        minimum=0.15,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )

            with gr.Row():
                clear = gr.ClearButton(
                    components=[prompt, negative_prompt, steps, seed, eta, cfg])
                submit = gr.Button(value='Submit')
    
            examples = gr.Examples(
                label="Quick Examples",
                examples=examples,
                inputs=[prompt, steps, 0, 0.3],
                outputs=[],
                cache_examples=False
            )

        with gr.Column():
            genImage = gr.Image(label='Generated Image', sources=['upload','clipboard'], interactive=True, type="filepath")
            seedTxt = gr.Markdown(label='Output Seed')
            preImage = gr.Image(label='First-Pass Image', sources=['upload','clipboard'], interactive=True, type="filepath")

    gr.Markdown(f'{article}')

    seed_state = gr.State()
    preimage_path_state = gr.State()

    submit.click(
        fn=first_pass_inference,
        inputs=[prompt, negative_prompt, steps, seed, eta, cfg, width, height, preImage],
        outputs=[genImage, seed_state, preimage_path_state]
    ).then(
        fn=high_res_inference,
        inputs=[genImage, prompt, negative_prompt, steps, seed_state, eta, cfg, hires_strength, width, height, preimage_path_state],
        outputs=[genImage]
    )
    randButton.click(fn=lambda: gr.Number(label="Random Seed", value=-1), show_progress=False, outputs=[seed])
    luckyButton.click(fn=lambda: gr.Number(label="Random Seed", value=newSeed()), show_progress=False, outputs=[seed])
    recycleButton.click(fn=lambda seedTxt: gr.Number(label="Random Seed", value=str2num(seedTxt)), show_progress=False, inputs=[seedTxt], outputs=[seed])

    genImage.upload(
        fn=get_params_from_image,
        inputs=[genImage],
        outputs=[prompt, negative_prompt, steps, seed, eta, cfg, hires_strength, width, height],
        show_progress=False
    )

    preImage.upload(
        fn=lambda img: img,  # Simply return the uploaded image
        inputs=[preImage],
        outputs=[preImage],
        show_progress=False
    )

demo.launch()

# TODO: add real-ESRGAN
# https://github.com/ai-forever/Real-ESRGAN
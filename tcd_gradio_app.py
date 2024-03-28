import random

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

from scheduling_tcd import TCDScheduler
from utils import save_image_with_geninfo, crc_hash, parse_params_from_image

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
"""

device = "mps"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

# original_forward = pipe.scheduler.forward

# def debug_forward(*args, **kwargs):
#     result = original_forward(*args, **kwargs)
#     if hasattr(result, "sigmas") and result.sigmas is not None:
#         print("Sigmas array:", result.sigmas)
#     return result

# pipe.scheduler.forward = debug_forward


def inference(prompt, negative_prompt="", steps=4, seed=-1, eta=0.3, cfg=0):
    if seed is None or seed == '' or seed == -1:
        seed = int(random.randrange(4294967294))
    print(f"prompt: {prompt}; negative: {negative_prompt}")
    print(f"seed: {seed}; steps: {steps}; eta: {eta}")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        eta=eta,
        generator=generator,
        height=1024,
        # width=768,
    ).images[0]
    d = {"seed": seed, "steps": steps, "eta": eta, "cfg": cfg, "prompt": prompt, "negative_prompt": negative_prompt}
    path = f"outputs/TCD_seed-{seed}_steps-{steps}_{crc_hash(repr(d))}.{output_format}"
    save_image_with_geninfo(image, str(d), path )
    return image


# Define style
title = "<h1>Trajectory Consistency Distillation</h1>"
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

def get_params_from_image(img) -> (str, str, int, int, float, float):
    p = parse_params_from_image(img)
    prompt = p.get('prompt','')
    negative_prompt = p.get('negative_prompt','')
    seed = p.get('seed',-1)
    steps = p.get('steps',4)
    eta = p.get('eta',0.3)
    cfg = p.get('cfg',1.0)
    return prompt, negative_prompt, steps, seed, eta, cfg

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
            with gr.Row():
                with gr.Column():
                    seed = gr.Number(label="Random Seed", value=-1)
                with gr.Column():
                    eta = gr.Slider(
                            label='Gamma',
                            minimum=0.,
                            maximum=1.,
                            value=0.3,
                            step=0.1,
                        )
                with gr.Column():
                    cfg = gr.Slider(
                            label='Guidance Scale (CFG)',
                            minimum=1,
                            maximum=3.,
                            value=1.,
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

    gr.Markdown(f'{article}')

    submit.click(
        fn=inference,
        inputs=[prompt, negative_prompt, steps, seed, eta, cfg],
        outputs=[genImage],
    )

    genImage.upload(
        fn=get_params_from_image,
        inputs=[genImage],
        outputs=[prompt, negative_prompt, steps, seed, eta, cfg],
        show_progress=False
    )

demo.launch()

import random

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

from scheduling_tcd import TCDScheduler
from utils import save_image_with_geninfo, crc_hash, parse_params_from_image, str2num
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
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# pipe = StableDiffusionXLPipeline.from_pretrained(
#     base_model_id,
#     torch_dtype=torch.float16,
#     variant="fp16"
# ).to(device)

base_model_dir = "/Users/dfl/sd/ComfyUI/models/checkpoints/"

base_model_id = "base/sd_xl_base_1.0"

pipe = StableDiffusionXLPipeline.from_single_file(
    f"{base_model_dir}{base_model_id}.safetensors",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)


pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = TCDScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     timestep_spacing="trailing",
# )


tcd_lora_id = ""

def refresh_pcm_steps(pcm_steps=0):
    global tcd_lora_id
    pipe.unload_lora_weights()
    if pcm_steps == 0:
        tcd_lora_id = "h1t/TCD-SDXL-LoRA"
        pipe.load_lora_weights(tcd_lora_id)
    elif pcm_steps == 5:
        weight_name = "pcm_sdxl_lcmlike_lora_converted.safetensors"
        tcd_lora_id = "wangfuyun/PCM_Weights"
        pipe.load_lora_weights(tcd_lora_id, weight_name=weight_name, subfolder="sdxl")
        # pipe.fuse_lora()
        tcd_lora_id += "/sdxl/" + weight_name # for exif params
    else:
        weight_name = f"pcm_sdxl_smallcfg_{2**pcm_steps}step_converted.safetensors"
        tcd_lora_id = "wangfuyun/PCM_Weights"
        pipe.load_lora_weights(tcd_lora_id, weight_name=weight_name, subfolder="sdxl")
        # pipe.fuse_lora()
        tcd_lora_id += "/sdxl/" + weight_name # for exif params
    print("swapping LoRA to ", tcd_lora_id)

refresh_pcm_steps()

# original_forward = pipe.scheduler.forward

# def debug_forward(*args, **kwargs):
#     result = original_forward(*args, **kwargs)
#     if hasattr(result, "sigmas") and result.sigmas is not None:
#         print("Sigmas array:", result.sigmas)
#     return result

# pipe.scheduler.forward = debug_forward


def newSeed() -> int:
    return int(random.randrange(4294967294))

def inference(prompt, negative_prompt="", steps=4, seed=-1, eta=0.3, cfg=0) -> (Image.Image, str):
    if seed is None or seed == '' or seed == -1:
        seed = newSeed()
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
    d = {
        "seed": seed, "steps": steps, "eta": eta, "cfg": cfg, "prompt": prompt, "negative_prompt": negative_prompt,
        "model": base_model_id, "lora": tcd_lora_id
    }
    path = f"outputs/TCD_seed-{seed}_steps-{steps}_{crc_hash(repr(d))}.{output_format}"
    save_image_with_geninfo(image, str(d), path )
    return image, f"seed: {seed}"
    

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
            pcm_steps = gr.Slider(
                label='PCM steps (2**N) [0=TCD, 5=LCM-like]',
                minimum=0,
                maximum=5,
                value=0,
                step=1,
            )
            pcm_steps.change(fn=refresh_pcm_steps, inputs=[pcm_steps])

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
                            step=0.05,
                        )
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
            seedTxt = gr.Markdown(label='Output Seed')

    gr.Markdown(f'{article}')

    submit.click(
        fn=inference,
        inputs=[prompt, negative_prompt, steps, seed, eta, cfg],
        outputs=[genImage, seedTxt],
    )

    randButton.click(fn=lambda: gr.Number(label="Random Seed", value=-1), show_progress=False, outputs=[seed])
    luckyButton.click(fn=lambda: gr.Number(label="Random Seed", value=newSeed()), show_progress=False, outputs=[seed])
    recycleButton.click(fn=lambda seedTxt: gr.Number(label="Random Seed", value=str2num(seedTxt)), show_progress=False, inputs=[seedTxt], outputs=[seed])

    genImage.upload(
        fn=get_params_from_image,
        inputs=[genImage],
        outputs=[prompt, negative_prompt, steps, seed, eta, cfg],
        show_progress=False
    )

demo.launch()

#!/usr/bin/env python
#patch 0.01 ()
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# ..

import os
import random
import uuid
from typing import Tuple

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTIONz= """## CANOPUS REALISM 🧒🏻


"""

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max

USE_TORCH_COMPILE = 0
ENABLE_CPU_OFFLOAD = 0

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("prithivMLmods/Canopus-Realism-LoRA", weight_name="Canopus-Realism-LoRA.safetensors", adapter_name="rlms")
    pipe.set_adapters("rlms")
    pipe.to("cuda")

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

DEFAULT_STYLE_NAME = "3840 x 2160"
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    if style_name in styles:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    else:
        p, n = styles[DEFAULT_STYLE_NAME]

    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

@spaces.GPU(duration=60, enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))

    positive_prompt, effective_negative_prompt = apply_style(style_name, prompt, negative_prompt)
    
    if not use_negative_prompt:
        effective_negative_prompt = ""  # type: ignore

    images = pipe(
        prompt=positive_prompt,
        negative_prompt=effective_negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=20,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil",
    ).images
    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

examples = [
    "A man dressed in sunglasses and brown jacket, in the style of cypherpunk, timeless beauty, exacting precision, uhd image, aleksandr deyneka, matte background, leather/hide  --ar 67:101 --v 5",
    "A black suit with black and white stripes, in the style of claire-obscure lighting, realistic genre scenes, light amber and violet, portraits with soft lighting, photo-realistic, cinematic lighting, hyper-realistic pop --ar 51:64 --v 5.2"
]

css = '''
.gradio-container{max-width: 545px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

def load_predefined_images():
    predefined_images = [
        "assets/3.png",
        "assets/4.png",
        "assets/5.png",
        "assets/6.png",
        "assets/7.png",
        "assets/8.png",
        "assets/9.png",
        "assets/10.png",
        "assets/11.png",
    ]
    return predefined_images



with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTIONz)  
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, preview=True, show_label=False)
    
    with gr.Accordion("Advanced options", open=False, visible=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
        negative_prompt = gr.Text(
            label="Negative prompt",
            lines=4,
            max_lines=6,
            value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
            placeholder="Enter a negative prompt",
            visible=True,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
        
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=3.0,
            )

        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Quality Style",
        )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=False,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            style_selection,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    # Adding a predefined gallery section
    gr.Markdown("### Generated Images")
    predefined_gallery = gr.Gallery(label="Generated Images", columns=3, show_label=False, value=load_predefined_images())
    gr.Markdown("**Disclaimer:**")
    gr.Markdown("This space provides realistic image generation, which works better for human faces and portraits. Realistic trigger works properly, better for photorealistic trigger words, close-up shots, face diffusion, male, female characters.")
   
    gr.Markdown("**Note:**")
    gr.Markdown("⚠️users are accountable for the content they generate and are responsible for ensuring it meets appropriate ethical standards.")
    
if __name__ == "__main__":
    demo.queue(max_size=40).launch()
    
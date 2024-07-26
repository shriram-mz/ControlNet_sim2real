
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from diffusers import UniPCMultistepScheduler
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import torch.nn as nn
from safetensors import safe_open

apply_uniformer = UniformerDetector()

def load_textual_inversion(model,embeddings_file_path):
    # tensors = {}
    embedding_matrix = None
    with safe_open(embeddings_file_path, framework="pt", device=0) as f:
        new_tokens = len(f.keys())

        for k in f.keys():
            # tensors[k] = f.get_tensor(k)
            #state_dict = torch.load(embeddings_file_path)
            if embedding_matrix is None:
                embedding_matrix = f.get_tensor(k)
            else :
               embedding_matrix = torch.cat((embedding_matrix,f.get_tensor(k))) 
            

    update_token_embeddings(model,new_tokens,embedding_matrix)

def update_token_embeddings(model,new_add_tokens , new_embedding_matrix):
    old_embeddings = model.cond_stage_model.transformer.text_model.embeddings.token_embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_num_tokens = old_num_tokens + new_add_tokens
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    new_embeddings.weight.data[n:,:] = new_embedding_matrix
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding = new_embeddings
    model.cond_stage_model.transformer.text_model.config.vocab_size = new_num_tokens
    model.cond_stage_model.transformer.text_model.vocab_size = new_num_tokens


model = create_model('/mnt/disks/data/sim2real/ControlNet/models/cldm_v15.yaml').cpu()
#model.load_state_dict(load_state_dict('/mnt/disks/data/sim2real/ControlNet/models/control_v11p_sd15_seg.pth', location='cuda'), strict=False)
# model.load_textual_inversion("/mnt/disks/data/sim2real/ControlNet/output_text_inversion/learned_embeds.safetensors")
load_textual_inversion(model,"/mnt/disks/data/sim2real/ControlNet/output_text_inversion_2/learned_embeds.safetensors")
model = model.cuda()
model.load_state_dict(load_state_dict('/mnt/disks/data/sim2real/ControlNet/textual_inversion_ControlNet/lightning_logs/version_0/checkpoints/epoch=64-step=112255.ckpt', location='cuda'), strict=False)
ddim_sampler = DDIMSampler(model)   


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)


        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Segmentation Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
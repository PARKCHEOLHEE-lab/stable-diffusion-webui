import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import os

url = "http://127.0.0.1:7860"

path = r"C:\BACKUP\Playdata\202105_lab\02.git\stable-diffusion-webui\viewport.png"
# path = r"C:\BACKUP\Playdata\202105_lab\02.git\stable-diffusion-webui\test2.jpg"
with open(path, "rb") as img_base64:
    input_image = base64.b64encode(img_base64.read())

width, height = Image.open(path).size

response_model_list = requests.get(f'{url}/controlnet/model_list')
(
    model_ip2p,
    model_shuffle,
    model_tile,
    model_depth,
    model_canny,
    model_inpaint,
    model_lineart,
    model_mlsd,
    model_normalbae,
    model_openpose,
    model_scribble,
    model_seg,
    model_softedge,
) = response_model_list.json()["model_list"]

response_module_list = requests.get(f'{url}/controlnet/module_list')
(
    module_none,
    module_canny,
    module_depth,
    module_depth_leres,
    module_depth_leres_pp,
    module_hed,
    module_hed_safe,
    module_mediapipe_face,
    module_mlsd,
    module_normal_map,
    module_openpose,
    module_openpose_hand,
    module_openpose_face,
    module_openpose_faceonly,
    module_openpose_full,
    module_clip_vision,
    module_color,
    module_pidinet,
    module_pidinet_safe,
    module_pidinet_sketch,
    module_pidinet_scribble,
    module_scribble_xdog,
    module_scribble_hed,
    module_segmentation,
    module_threshold,
    module_depth_zoe,
    module_normal_bae,
    module_oneformer_coco,
    module_oneformer_ade20k,
    module_lineart,
    module_lineart_coarse,
    module_lineart_anime,
    module_lineart_standard,
    module_shuffle,
    module_tile_resample,
    module_invert,
    module_lineart_anime_denoise,
    module_reference_only,
    module_reference_adain,
    module_reference_adain_attn,
    module_inpaint,
    module_inpaint_only,
    module_inpaint_only_lama,
    module_tile_colorfix,
module_tile_colorfix_sharp
) = response_module_list.json()["module_list"]

prompt = """
    Interior view, 
    Large window of side wall with trees, 
    Black sofa, 
    Stools, 
    Wood table, 
    White rug, 
    Drawer
"""

payload = {
    "prompt": prompt,
    "negative_prompt": "",
    "resize_mode": 0,
    "denoising_strength": 0.75,
    "mask_blur": 36,
    "inpainting_fill": 0,
    "inpaint_full_res": "true",
    "inpaint_full_res_padding": 72,
    "inpainting_mask_invert": 0,
    "initial_noise_multiplier": 1,
    "seed": -1,
    "sampler_name": "Euler a",
    "batch_size": 1,
    "steps": 1,
    "cfg_scale": 7,
    "width": width,
    "height": height,
    "restore_faces": "false",
    "tiling": "false",
    "alwayson_scripts": {
        "ControlNet": {
        "args": [
            {
            "enabled": "true",
            "input_image": input_image.decode(),
            "module": module_pidinet_scribble,
            "model": model_scribble,
            },
        ]
        }
    }
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
r = response.json()

for i in r['images'][:-1]:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    # png_payload = {
    #     "image": "data:image/png;base64," + i
    # }
    # response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    # pnginfo = PngImagePlugin.PngInfo()
    # pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output.png')
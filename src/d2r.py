import os
import json
import Rhino
import base64
import urllib2
import scriptcontext as sc
import System.Drawing.Imaging as Imaging


class D2R:
    """Convert Draft to Rendered using `stable diffusion webui` API"""
    
    def __init__(
        self, prompt, width=512, height=512, local_url="http://127.0.0.1:7860"
    ):
        self.prompt = prompt
        self.width = width
        self.height = height
        self.local_url = local_url
        
        try:
            (   # Unpack model names string
                self.model_ip2p,
                self.model_shuffle,
                self.model_tile,
                self.model_depth,
                self.model_canny,
                self.model_inpaint,
                self.model_lineart,
                self.model_mlsd,
                self.model_normalbae,
                self.model_openpose,
                self.model_scribble,
                self.model_seg,
                self.model_softedge,
            ) = self._get_model_list(local_url)
            
            
            (   # Unpack module names string
                self.module_none,
                self.module_canny,
                self.module_depth,
                self.module_depth_leres,
                self.module_depth_leres_pp,
                self.module_hed,
                self.module_hed_safe,
                self.module_mediapipe_face,
                self.module_mlsd,
                self.module_normal_map,
                self.module_openpose,
                self.module_openpose_hand,
                self.module_openpose_face,
                self.module_openpose_faceonly,
                self.module_openpose_full,
                self.module_clip_vision,
                self.module_color,
                self.module_pidinet,
                self.module_pidinet_safe,
                self.module_pidinet_sketch,
                self.module_pidinet_scribble,
                self.module_scribble_xdog,
                self.module_scribble_hed,
                self.module_segmentation,
                self.module_threshold,
                self.module_depth_zoe,
                self.module_normal_bae,
                self.module_oneformer_coco,
                self.module_oneformer_ade20k,
                self.module_lineart,
                self.module_lineart_coarse,
                self.module_lineart_anime,
                self.module_lineart_standard,
                self.module_shuffle,
                self.module_tile_resample,
                self.module_invert,
                self.module_lineart_anime_denoise,
                self.module_reference_only,
                self.module_reference_adain,
                self.module_reference_adain_attn,
                self.module_inpaint,
                self.module_inpaint_only,
                self.module_inpaint_only_lama,
                self.module_tile_colorfix,
                self.module_tile_colorfix_sharp
            ) = self._get_module_list(local_url)
        
        except Exception as e:
            print(e)
            print("You have no connection to local API server")
        
    def _get_model_list(self, local_url):
        """Get model names string list""" 
        request = urllib2.Request(
            "{url}/controlnet/model_list".format(url=local_url)
        )
        
        response = urllib2.urlopen(request)
        return json.loads(response.read())["model_list"]
    
    def _get_module_list(self, local_url):
        """Get module names string list""" 
        
        request = urllib2.Request(
            "{url}/controlnet/module_list".format(url=local_url)
        )
        
        response = urllib2.urlopen(request)
        
        return json.loads(response.read())["module_list"]
        
    def _get_decoded_image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data)
            decoded_image = base64_data.decode('utf-8')
            return decoded_image
            
    def _save_base64_to_png(self, base64_string, save_path):
        """
            Convert and save base64 image to png format image
        """
        
        decoded_image = base64.b64decode(base64_string)
    
        with open(save_path, 'wb') as file:
            file.write(decoded_image)
        
    def capture_activated_viewport(
        self, save_name="draft.png", return_size=False
    ):
        """
            Capture and save the currently activated viewport 
            to the location of the current *.gh file path
        """
        
        save_path = os.path.join(CURRENT_DIR, save_name)
        
        viewport = Rhino.RhinoDoc.ActiveDoc.Views.ActiveView.CaptureToBitmap()
        viewport.Save(save_path, Imaging.ImageFormat.Png)
        
        if return_size:
            return save_path, viewport.Size
        
        return save_path
    
    def render(self, image_path, seed=-1, steps=20, draft_size=None):
        payload = {
            "prompt": self.prompt,
            "negative_prompt": "",
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "mask_blur": 36,
            "inpainting_fill": 0,
            "inpaint_full_res": "true",
            "inpaint_full_res_padding": 72,
            "inpainting_mask_invert": 0,
            "initial_noise_multiplier": 1,
            "seed": seed,
            "sampler_name": "Euler a",
            "batch_size": 1,
            "steps": steps,
            "cfg_scale": 4,
            "width": self.width if draft_size is None else draft_size.Width,
            "height": self.height if draft_size is None else draft_size.Height,
            "restore_faces": "false",
            "tiling": "false",
            "alwayson_scripts": {
                "ControlNet": {
                "args": [
                    {
                        "enabled": "true",
                        "input_image": self._get_decoded_image_to_base64(image_path),
                        "module": self.module_pidinet_scribble,
                        "model": self.model_scribble,
                        "processor_res": 1024,
                    },
                ]
                }
            }
        }
        
        request = urllib2.Request(
            url=self.local_url + "/sdapi/v1/txt2img", 
            data=json.dumps(payload), 
            headers={'Content-Type': 'application/json'}
        )

        try:
            response = urllib2.urlopen(request)
            response_data = response.read()
            
            rendered_save_path = os.path.join(CURRENT_DIR, "rendered.png")
            converted_save_path = os.path.join(CURRENT_DIR, "converted.png")
            
            response_data_jsonify = json.loads(response_data)
            
            used_seed = json.loads(response_data_jsonify["info"])["seed"]
            used_params = response_data_jsonify["parameters"]
            
            for ii, image in enumerate(response_data_jsonify["images"]):
                
                if ii == len(response_data_jsonify["images"]) - 1:
                    self._save_base64_to_png(image, converted_save_path)
                else:
                    self._save_base64_to_png(image, rendered_save_path)

            return (
                rendered_save_path, 
                converted_save_path,
                used_seed, 
                used_params
            )
        
        except urllib2.HTTPError as e:
            print("HTTP Error:", e.code, e.reason)
            response_data = e.read()
            print(response_data)
            
            return None



if __name__ == "__main__":
    CURRENT_FILE = sc.doc.Path
    CURRENT_DIR = "\\".join(CURRENT_FILE.split("\\")[:-1])
    
    prompt = (
        """
            Interior view with sunlight,
            Curtain wall with city view
            Colorful Sofas,
            Cushions on the sofas
            Transparent glass Table,
            Fabric stools,
            Some flower pots
        """
    )
    
    d2r = D2R(prompt=prompt)
    
    draft, draft_size = d2r.capture_activated_viewport(return_size=True)
    rendered, converted, seed, params = d2r.render(
        draft, seed=-1, steps=50, draft_size=draft_size
    )

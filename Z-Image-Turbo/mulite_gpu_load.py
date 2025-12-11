from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor, PreTrainedModel, AutoTokenizer
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel, QwenImageEditPipeline, QwenImagePipeline, QwenImageEditPlusPipeline, ZImagePipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights, load_checkpoint_and_dispatch
import torch
import os
from typing import Optional, Dict, Any

from PIL import Image


class MultiGPUModel:

    def __init__(self,  
    local_files_only=True, 
    trust_remote_code=True, 
    dtype=torch.bfloat16
    ):


        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
    
    def load_tokenizer(self, tokenizer_path):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
        )
        return tokenizer
    
    def load_z_tokenizer(self, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
        )
        return tokenizer
    
    def load_processor(self, processor_path):

        processor = Qwen2VLProcessor.from_pretrained(
            processor_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
        )

        return processor
    
    def load_text_encoder(self, text_encoder_path, device):

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path, 
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
        ).to(device)

        return text_encoder
    
    def load_vae(self, vae_path, device):

        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path, 
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
            ).to(device)

        return vae
    
    def load_scheduler(self, scheduler_path):

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            scheduler_path, 
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code
        )
        return scheduler
    
    def load_transformer(self, transformer_path, device_map_gpu):

        with init_empty_weights():
            empty_model = QwenImageTransformer2DModel.from_config(
                transformer_path + "/config.json"
            )
        device_map = infer_auto_device_map(
            empty_model,
            max_memory= device_map_gpu,
            no_split_module_classes=["QwenImageTransformerBlock"],
            dtype=self.dtype, 
        )
        transformer = load_checkpoint_and_dispatch(
            empty_model,
            transformer_path,
            device_map=device_map,
            dtype=self.dtype,
        )
        return transformer
    
    def load_QwenImageEditPipeline(self, 
        model_path,     
        device_map={
            "transformer":{0: "35GB", 1: "35GB", 2: "35GB"},
            "vae_and_text_encoder":"cuda:2",
        }
        ):
        tokenizer = self.load_tokenizer(model_path + "/tokenizer")
        processor = self.load_processor(model_path + "/processor")
        text_encoder = self.load_text_encoder(model_path + "/text_encoder", device_map["vae_and_text_encoder"])
        vae = self.load_vae(model_path + "/vae", device_map["vae_and_text_encoder"])
        scheduler = self.load_scheduler(model_path + "/scheduler")
        transformer = self.load_transformer(model_path + "/transformer", device_map["transformer"])

        pipeline = QwenImageEditPlusPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler
        )
        
        return pipeline
    
    def laod_QwenImagePipeline(self, 
    model_path,
    device_map={
            "transformer":{0: "35GB", 1: "35GB", 2: "35GB"},
            "vae_and_text_encoder":"cuda:2",
        }
    ):
        scheduler = self.load_scheduler(model_path + "/scheduler")
        vae = self.load_vae(model_path + "/vae", device_map["vae_and_text_encoder"])
        text_encoder = self.load_text_encoder(model_path + "/text_encoder", device_map["vae_and_text_encoder"])
        tokenizer = self.load_tokenizer(model_path + "/tokenizer")
        transformer = self.load_transformer(model_path + "/transformer", device_map["transformer"])

        pipeline = QwenImagePipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler
        )

        return pipeline

    def load_Image_and_Edit_Pipeline(self, 
    Image_model_path, 
    Edit_model_path,
    device_map={
            "transformer0":{0: "35GB", 1: "35GB", 2: "35GB"},
            "transformer1":{5: "35GB", 6: "35GB", 7: "35GB"},
            "vae_and_text_encoder":"cuda:2",
        }
    ):
        scheduler = self.load_scheduler(Edit_model_path + "/scheduler")
        vae = self.load_vae(Image_model_path + "/vae", device_map["vae_and_text_encoder"])
        text_encoder = self.load_text_encoder(Image_model_path + "/text_encoder", device_map["vae_and_text_encoder"])
        tokenizer = self.load_tokenizer(Image_model_path + "/tokenizer")

        img_transformer = self.load_transformer(Image_model_path + "/transformer", device_map["transformer0"])
        edit_transformer = self.load_transformer(Edit_model_path + "/transformer", device_map["transformer1"])
        edit_processor = self.load_processor(Edit_model_path + "/processor")


        image_pipeline = QwenImagePipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=img_transformer,
            scheduler=scheduler
        )

        edit_pipeline = QwenImageEditPlusPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=edit_processor,
            transformer=edit_transformer,
            scheduler=scheduler
        )

        return image_pipeline, edit_pipeline
    

def resize_to_multiple_of_16(image):
    """将图片调整为能被16整除的尺寸"""
    width, height = image.size
    
    # 计算能被16整除的新尺寸
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16
    
    # 如果原尺寸已经是16的倍数，则不需要调整
    if width == new_width and height == new_height:
        return image
    
    # 如果原尺寸不是16的倍数但不想直接截断，也可以向上取整
    # new_width = ((width + 15) // 16) * 16
    # new_height = ((height + 15) // 16) * 16
    
    # 调整图片尺寸
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image
def edit_create_prompt(image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }
    return inputs

def image_create_prompt(prompt):
    inputs = {
        "prompt": prompt,
        "negative_prompt": " ",
        "generator": torch.manual_seed(0),
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
    }
    return inputs

if __name__ == "__main__":
    model_path = "/data/zh/model/Qwen/Qwen-Image-Edit"
    img_model_path = "/data/zh/model/Qwen-Image"
    multi_gpu_model = MultiGPUModel()
    image_pipeline, edit_pipeline = multi_gpu_model.load_Image_and_Edit_Pipeline(img_model_path, model_path)
    
    inputs = edit_create_prompt("/data/zh/image.jpg", "背景改为蓝天白云")
    image_inputs = image_create_prompt("一个穿着红色衣服的女孩")
    with torch.inference_mode():
        output = edit_pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("test001.png")
        print("image saved at", os.path.abspath("test001.png"))

        output_image = image_pipeline(**image_inputs)
        output_image = output_image.images[0]
        output_image.save("test002.png")
        print("image saved at", os.path.abspath("test002.png"))
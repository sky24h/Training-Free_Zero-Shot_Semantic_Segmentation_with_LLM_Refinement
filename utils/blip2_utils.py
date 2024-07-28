import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from .env_utils import get_device, low_vram_mode

device = get_device()

blip2_model_id = "Salesforce/blip2-opt-2.7b"  # or replace with your local model path
blip2_precision = torch.float16

# Load BLIP2 model and processor from HuggingFace
blip2_processor = Blip2Processor.from_pretrained(blip2_model_id)
if low_vram_mode:
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_model_id,
        torch_dtype         = blip2_precision,
        device_map          = device,
        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if low_vram_mode else None,
    ).eval()
else:
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(blip2_model_id, torch_dtype=blip2_precision, device_map=device).eval()


def blip2_caption(raw_image):
    # unconditional image captioning
    inputs  = blip2_processor(raw_image, return_tensors="pt")
    inputs  = inputs.to(device=device, dtype=blip2_precision)
    out     = blip2_model.generate(**inputs)
    caption = blip2_processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    from PIL import Image

    # Test the RAM++ model
    image_path = os.path.join(os.path.dirname(__file__), "../sources/test_imgs/1.jpg")
    image = Image.open(image_path)
    result = blip2_caption(image)
    print(result)

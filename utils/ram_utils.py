import os
import torch
from PIL import Image
from .env_utils import get_device, low_vram_mode

device = get_device()

pretrained_models_folder = os.path.join(os.path.dirname(__file__), "../pretrained-models")


# RAM++
from ram.models import ram_plus
from ram import get_transform, inference_ram

ram_ckpt = os.path.join(pretrained_models_folder, "checkpoints/ram_plus_swin_large_14m.pth")
ram_precision = torch.float16


def ram_init():
    image_size = 384
    transform = get_transform(image_size=image_size)
    #######load model#######
    model = ram_plus(pretrained=ram_ckpt, image_size=image_size, vit="swin_l")
    model = model.to(device=device, dtype=ram_precision)
    model.eval()
    print("RAM++ model loaded")
    return model, transform


# Initialize the model when importing the module
ram_model, ram_transform = ram_init()


def _inference(image_pil):
    image = ram_transform(image_pil).unsqueeze(0)
    image = image.to(device=device, dtype=ram_precision)
    res = inference_ram(image, ram_model)
    result = res[0].replace(" | ", ", ")
    return result


def _split_large_image(image_pil):
    size = image_pil.size
    print("Image size is too large, split into smaller patches")
    # Split the image into 4 patches
    patches = []
    patch_size = (size[0] // 2, size[1] // 2)
    for i in range(2):
        for j in range(2):
            left   = i * patch_size[0]
            top    = j * patch_size[1]
            right  = left + patch_size[0]
            bottom = top + patch_size[1]
            patch  = image_pil.crop((left, top, right, bottom))
            patches.append(patch)
    return patches


def ram_inference(image_pil: Image.Image):
    size = image_pil.size
    if size[0] > 640 or size[1] > 640:
        patches = _split_large_image(image_pil)
        while any(patch.size[0] > 640 or patch.size[1] > 640 for patch in patches):
            patches = [_split_large_image(patch) for patch in patches]
            patches = [patch for sublist in patches for patch in sublist]
        # Inference on each patch
        results = []
        for patch in patches:
            result = _inference(patch)
            results.extend(result.split(", "))
        results = list(set(results))
        # Combine the results
        final_result = ", ".join(results)
        return final_result
    else:
        print("Image size is small enough for inference")
        return _inference(image_pil)


if __name__ == "__main__":
    # Test the RAM++ model
    image_path = os.path.join(os.path.dirname(__file__), "../sources/test_imgs/1.jpg")
    image = Image.open(image_path)
    result = ram_inference(image)
    print(result)

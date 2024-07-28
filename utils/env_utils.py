# Avoid multiple imports of the same module. Use this to import the module only once.
# Also, ensure that the device and pretrained models folder are consistent across the project.

import os
import torch

global low_vram_mode
low_vram_mode = False


def use_lower_vram():
    global low_vram_mode
    low_vram_mode = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pretrained_models_folder():
    return os.path.join(os.path.dirname(__file__), "../pretrained-models")


def download_pretrained_models():
    pretrained_models_folder = get_pretrained_models_folder()
    # hard-coded download links
    groundingdino_link = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
    sam_link           = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    ram_link           = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
    groundingdino_ckpt = os.path.join(pretrained_models_folder, "checkpoints/groundingdino_swint_ogc.pth")
    sam_ckpt           = os.path.join(pretrained_models_folder, "checkpoints/sam_hq_vit_l.pth")
    ram_ckpt           = os.path.join(pretrained_models_folder, "checkpoints/ram_plus_swin_large_14m.pth")

    # download pretrained models if not exists
    if not os.path.exists(groundingdino_ckpt):
        os.system(f"wget -O {groundingdino_ckpt} {groundingdino_link}")
    if not os.path.exists(sam_ckpt):
        os.system(f"wget -O {sam_ckpt} {sam_link}")
    if not os.path.exists(ram_ckpt):
        os.system(f"wget -O {ram_ckpt} {ram_link}")


# download pretrained models when imported
download_pretrained_models()

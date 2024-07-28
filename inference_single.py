import os
import cv2
from PIL import Image
from omegaconf import OmegaConf
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, help="path to config file")
    parser.add_argument("--input_path", default=None, required=True, help="path to input image")
    parser.add_argument("--draw_bbox", default=False, action="store_true", help="draw bounding boxes and points")
    parser.add_argument("--use_lower_vram", default=False, action="store_true", help="use low vram mode")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # set up environment
    from utils.env_utils import get_device, set_random_seed, use_lower_vram
    from utils.timer_utils import Timer

    set_random_seed(1024)
    device = get_device()
    timer = Timer(reset=True)

    if args.use_lower_vram:
        use_lower_vram()

    # import functions
    from utils.labels_utils import Labels
    from utils.ram_utils import ram_inference
    from utils.blip2_utils import blip2_caption
    from utils.llms_utils import pre_refinement, make_prompt, init_model
    from utils.grounded_sam_utils import run_grounded_sam, draw_bboxes

    # get config
    box_threshold     = config.box_threshold
    text_threshold    = config.text_threshold
    iou_threshold     = config.iou_threshold
    example_size      = config.example_size
    output_dir        = os.path.join(os.path.dirname(__file__), "outputs_single", config.Name)
    os.makedirs(output_dir, exist_ok=True)

    # init labels and llm prompt
    L = Labels(config=config)
    llm = init_model(config.llm_model, api_key=config.api_key)

    # get input images, they are expected either in jpg or png format
    input_path = args.input_path
    basename = os.path.basename(input_path).split(".")[0]

    system_prompt = make_prompt(", ".join(L.LABELS), None, None, llm=llm)

    image_pil = Image.open(input_path).convert("RGB")
    labels_ram = ram_inference(image_pil) + ": " + blip2_caption(image_pil)
    converted_labels, llm_output = pre_refinement([labels_ram], system_prompt, llm=llm)
    labels_llm = L.check_labels(converted_labels)[0]
    print("labels_ram: ", labels_ram)
    print("llm_output: ", llm_output)
    print("labels_llm: ", labels_llm)

    # run sam
    label_res, bboxes, output_labels, output_prob_maps, output_points = run_grounded_sam(
        input_image    = {"image": image_pil, "mask": None},
        text_prompt    = labels_llm,
        box_threshold  = box_threshold,
        text_threshold = text_threshold,
        iou_threshold  = iou_threshold,
        LABELS         = L.LABELS,
        IDS            = L.IDS,
        llm            = llm,
        timer          = timer,
    )

    # draw mask and save image
    image_ori = cv2.imread(input_path)
    save_path = os.path.join(output_dir, basename + "_result.jpg")
    ours = L.draw_mask(label_res, image_ori, print_label=True, tag="Ours")
    cv2.imwrite(save_path, ours, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # draw bboxes and save image
    if args.draw_bbox:
        ours_bboxes = draw_bboxes(ours.copy(), output_labels, bboxes, output_points, output_prob_maps)
        save_path_bboxes = os.path.join(output_dir, basename + "_bboxes.jpg")
        cv2.imwrite(save_path_bboxes, ours_bboxes, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Saved to: ", save_path_bboxes, "\n") if args.draw_bbox else print("Saved to: ", save_path, "\n")

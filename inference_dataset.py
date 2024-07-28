import os
import cv2
import glob
import random
import shutil
import traceback
import numpy as np
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from termcolor import colored
from omegaconf import OmegaConf
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, help="path to config file")
    parser.add_argument("--draw_bbox", default=False, action="store_true", help="draw bounding boxes and points")
    parser.add_argument("--debug", default=False, action="store_true", help="using only 5% of the images")
    parser.add_argument("--reset", default=False, action="store_true", help="rm all outputs before running")
    parser.add_argument("--use_lower_vram", default=False, action="store_true", help="use low vram mode")
    args   = parser.parse_args()
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
    output_dir        = os.path.join(os.path.dirname(__file__), "outputs", config.Name)
    output_dir_ram    = os.path.join(output_dir, "ram")
    output_dir_llm    = os.path.join(output_dir, "llm")
    output_dir_masks  = os.path.join(output_dir, "masks")
    output_dir_images = os.path.join(output_dir, "images")
    output_dir_bboxes = os.path.join(output_dir, "bboxes") if args.draw_bbox else ""
    if args.reset:
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        # remove only the final outputs
        shutil.rmtree(output_dir_masks, ignore_errors=True) if os.path.exists(output_dir_masks) else None
        shutil.rmtree(output_dir_images, ignore_errors=True) if os.path.exists(output_dir_images) else None
        shutil.rmtree(output_dir_bboxes, ignore_errors=True) if os.path.exists(output_dir_bboxes) else None

    # get flags for abalation study, wo_blip2, wo_pre, wo_post, if not provided, set to False
    wo_blip2 = config.wo_blip2 if hasattr(config, "wo_blip2") else False
    wo_pre   = config.wo_pre if hasattr(config, "wo_pre") else False
    wo_post  = config.wo_post if hasattr(config, "wo_post") else False

    # init labels and llm prompt
    L = Labels(config=config)
    try:
        llm = init_model(config.llm_model, api_key=config.api_key)
    except: # noqa
        print("Failed to init llm model, skipping...")
        traceback.print_exc()
        llm = None

    # create output dirs
    os.makedirs(output_dir_ram, exist_ok=True)
    os.makedirs(output_dir_llm, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_bboxes, exist_ok=True) if args.draw_bbox else None

    # get input images, they are expected either in jpg or png format
    input_dir = config.input_dir
    all_imgs  = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

    # in case to compare during inference
    gt_dir = config.gt_dir if hasattr(config, "gt_dir") else None

    # in case to find all pre-defined labels without LLMs
    mode = config.Mode if hasattr(config, "Mode") else None
    skip_ram_llm = False
    use_gt = False
    if mode is not None:
        if mode == "FindALL":
            skip_ram_llm = True
        elif mode == "GT":
            skip_ram_llm = True
            use_gt = True
        else:
            raise NotImplementedError
    else:
        skip_ram_llm = False
    print("skip_ram_llm", skip_ram_llm, "use_gt", use_gt)

    if args.debug:
        all_imgs = natsorted(all_imgs)[:len(all_imgs) // 20]
        print(colored("Using only 5% of the images", "red"))
    else:
        all_imgs = natsorted(all_imgs)
        print(colored("Using all images", "green"))
    print("Total numbers of images:", len(all_imgs))

    mutual_labels = ""
    if not skip_ram_llm:
        if example_size > 0:
            example_list  = random.sample(all_imgs, example_size)
            example_bases = [img_path.replace(input_dir, "").replace("/", "_").split(".")[0][1:] for img_path in example_list]
            example_list  = [Image.open(img_path).convert("RGB") for img_path in example_list]
            example_rams  = [ram_inference(image_pil) for image_pil in example_list]
            example_gts   = [L.find_gt_labels(np.load(os.path.join(gt_dir, basename + ".npy"), allow_pickle=True)) for basename in example_bases] # type: ignore
            example_gts   = [", ".join(labels_llm) for labels_llm in example_gts]
            mutual_labels = set(example_gts[0].split(", "))
            for labels in example_gts[1:]:
                mutual_labels.intersection_update(set(labels.split(", ")))
            mutual_labels = ", ".join(list(mutual_labels))
            print("Mutual labels:", mutual_labels)
        else:
            example_rams = None
            example_gts  = None
        system_prompt = make_prompt(", ".join(L.LABELS), example_rams, example_gts, llm=llm)

    batch_size = 1
    assert batch_size == 1, "inference with batch size > 1 will cause decrease in performance or even error, due to the ability of the LLMs."
    for i_batch in tqdm(range(0, len(all_imgs), batch_size)):
        try:
            timer.start()
            # input images
            img_paths  = all_imgs[i_batch : i_batch + batch_size]
            image_pils = [Image.open(img_path).convert("RGB") for img_path in img_paths]
            basenames  = [img_path.replace(input_dir, "").replace("/", "_").split(".")[0][1:] for img_path in img_paths]

            # init lists
            labels_rams = []
            llm_outputs = []
            labels_llms = []
            if gt_dir is not None:
                labels_gts = [L.find_gt_labels(np.load(os.path.join(gt_dir, basename + ".npy"), allow_pickle=True)) for basename in basenames]
                labels_gts = [", ".join(labels_gt) for labels_gt in labels_gts]
            else:
                labels_gts = [None for _ in range(len(basenames))]

            if not skip_ram_llm:
                alreay_done = True
                for basename in basenames:
                    # check if alreay done
                    save_path_ram = os.path.join(output_dir_ram, basename + ".txt")
                    # print("save_path_ram: ", save_path_ram)
                    if os.path.exists(save_path_ram):
                        print("Already done in RAM++")
                        labels_rams.append(open(save_path_ram, "r").read())
                    else:
                        alreay_done = False
                if not alreay_done:
                    if wo_blip2:
                        # do not use BLIP2, for ablation study
                        print("wo_blip2 is True, for ablation study")
                        labels_rams = [ram_inference(image_pil) for image_pil in image_pils]
                    else:
                        labels_rams = [ram_inference(image_pil) + ": " + blip2_caption(image_pil) for image_pil in image_pils]
                    # print("labels_rams: ", labels_rams)
                    for labels_ram, basename in zip(labels_rams, basenames):
                        with open(os.path.join(output_dir_ram, basename + ".txt"), "w") as f:
                            f.write(labels_ram)
                timer.check("ram_inference")

                llm_success = False
                try:
                    last_save_path_llm = os.path.join(output_dir_llm, basenames[-1] + ".txt")
                    # print("last_save_path_llm: ", last_save_path_llm)
                    if os.path.exists(last_save_path_llm):
                        print("Already done in llm")
                        for basename in basenames:
                            save_path_llm = os.path.join(output_dir_llm, basename + ".txt")
                            labels_llms.append(open(save_path_llm, "r").read())
                        llm_outputs = [None for _ in range(len(basenames))]
                    else:
                        if wo_pre:
                            # using output from ram++ directly, for ablation study
                            llm_outputs = labels_rams
                            labels_llms = labels_rams
                        else:
                            print("Sending to llm...")
                            converted_labels, llm_outputs = pre_refinement(labels_rams, system_prompt, llm=llm)
                            print("llm_outputs: ", llm_outputs)
                            labels_llms = L.check_labels(converted_labels)
                    # print("caption_llms: ", caption_llms)
                    for labels_llm, basename in zip(labels_llms, basenames):
                        with open(os.path.join(output_dir_llm, basename + ".txt"), "w") as f:
                            f.write(labels_llm)

                    if len(labels_rams) == len(labels_llms):
                        llm_success = True
                    else:
                        print("number of labels_rams != number of labels_llms, retrying...")
                        continue
                except: # noqa
                    print("Error in llm, retrying...")
                    traceback.print_exc()
                    continue

                if llm_success is False:
                    print("Failed to run llm, skipping...")
                    continue
                timer.check("llm call")
            else:
                labels_rams = [None for _ in range(len(basenames))]
                llm_outputs = [None for _ in range(len(basenames))]
                if use_gt:
                    if gt_dir is not None:
                        # skip RAM and llm, use GT labels
                        labels_llms = labels_gts
                    else:
                        raise NotImplementedError("GT is not provided")
                else:
                    # skip RAM and llm, find all labels
                    labels_llms = [", ".join(L.LABELS[1:]) for _ in range(len(basenames))]
        except Exception as e:
            print("Error! skipping... ", e)
            traceback.print_exc()
            continue
        # print("lens:", len(img_paths), len(image_pils), len(basenames), len(labels_rams), len(llm_outputs), len(labels_llms), len(labels_gts))

        skip_sam = False
        for img_path, image_pil, basename, labels_ram, llm_output, labels_llm, labels_gt in zip(img_paths, image_pils, basenames, labels_rams, llm_outputs, labels_llms, labels_gts):
            if not skip_ram_llm:
                print("labels_ram: ", labels_ram)
                print("llm_output: ", llm_output)
                print("labels_llm: ", labels_llm)
                print("labels_gt : ", labels_gt)
                # print missing labels in labels_llm compared to labels_gt
                if labels_gt is not None:
                    labels_gt_ = labels_gt.split(", ")
                    labels_llm_ = labels_llm.split(", ")
                    missing_labels = [label for label in labels_gt_ if label not in labels_llm_]
                    if len(missing_labels) > 0:
                        print(colored("missing_labels: ", "red"), missing_labels)
                    else:
                        print(colored("All labels are found!", "green"))
                print("")
            if labels_llm == "":
                print("Empty label from llm, skipping...")
                continue
            if skip_sam:
                print("Skip SAM")
                continue
            label_res = None
            try:
                for _ in range(3):
                    try:
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
                            wo_post        = (skip_ram_llm or wo_post),
                        )
                        break
                    except: # noqa
                        print("Failed to run SAM, retrying...")
                        traceback.print_exc()
                        continue
                if label_res is None:
                    print("Failed to run SAM, skipping...")
                    # save empty mask
                    empty_label = np.array(Image.new("L", image_pil.size, color=0))
                    np.save(os.path.join(output_dir_masks, basename + ".npy"), empty_label)
                    continue
                np.save(os.path.join(output_dir_masks, basename + ".npy"), label_res)

                # draw mask and save image
                image_ori = cv2.imread(img_path)
                save_path = os.path.join(output_dir_images, basename + ".jpg")
                ours = L.draw_mask(label_res, image_ori, print_label=True, tag = "Ours")
                if gt_dir is not None:
                    label_gt = np.load(os.path.join(gt_dir, basename + ".npy"), allow_pickle=True)
                    gt = L.draw_mask(label_gt, image_ori, print_label=True, tag = "GT")
                    res = cv2.hconcat([ours, gt])
                else:
                    res = ours
                cv2.imwrite(save_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # draw bboxes and save image
                if args.draw_bbox:
                    ours_bboxes = draw_bboxes(ours.copy(), output_labels, bboxes, output_points, output_prob_maps)
                    if gt_dir is not None:
                        ours_bboxes = cv2.hconcat([ours_bboxes, gt])
                    save_path_bboxes = os.path.join(output_dir_bboxes, basename + ".jpg")
                    cv2.imwrite(save_path_bboxes, ours_bboxes, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print("Saved to: ", save_path_bboxes, "\n") if args.draw_bbox else print("Saved to: ", save_path, "\n")
            except Exception as e:
                print("Error!", e)
                traceback.print_exc()
                continue
        timer.stop()

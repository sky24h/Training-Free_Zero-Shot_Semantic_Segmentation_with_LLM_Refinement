import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image, ImageFont
import traceback

# environment variables and paths
from .env_utils import get_device, get_pretrained_models_folder

device = get_device()
pretrained_models_folder = get_pretrained_models_folder()
groundingdino_ckpt = os.path.join(pretrained_models_folder, "checkpoints/groundingdino_swint_ogc.pth")
sam_ckpt = os.path.join(pretrained_models_folder, "checkpoints/sam_hq_vit_l.pth")

# segment anything
from segment_anything import build_sam_vit_l, SamPredictor

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

font_family = os.path.join(os.path.dirname(__file__), "Arial.ttf")
font_size = 24
font = ImageFont.truetype(font_family, font_size)

from .llms_utils import post_refinement


def draw_bboxes(ours_bboxes, output_labels, bboxes, output_points, output_prob_maps):
    # draw bboxes on the image
    for label, bbox in zip(output_labels, bboxes):
        bbox = bbox.cpu().numpy()
        bbox = [int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))]
        # print("label, bbox", label, bbox)
        cv2.rectangle(ours_bboxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # caption inside the bbox, below the top left corner 20 pixels
        cv2.putText(ours_bboxes, label, (bbox[0], bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    try:
        for points in output_points:
            for point in points:
                # draw a cross on the point
                cv2.drawMarker(ours_bboxes, (int(point[0]), int(point[1])), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
    except: # noqa
        pass

    # Draw the probability maps
    # if output_prob_maps is not None:
    #     output_prob_maps = np.concatenate(output_prob_maps, axis=1)
    #     ours_bboxes = np.concatenate([output_prob_maps, ours_bboxes], axis=1)
    return ours_bboxes


def transform_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def _load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    model.load_state_dict(clean_state_dict(torch.load(model_checkpoint_path, map_location="cpu")["model"]), strict=False)
    return model.to(device=device).eval()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt  = boxes.clone()
    filt_mask   = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt  = boxes_filt[filt_mask]  # num_filt,  4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized  = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
    return boxes_filt, torch.Tensor(scores), pred_phrases


def postprocess_masks(input_masks, input_pred_phrases):
    input_masks_ = input_masks.cpu().numpy().transpose(0, 2, 3, 1).copy()
    output_masks = input_masks.cpu().numpy().transpose(0, 2, 3, 1).copy()
    for i in range(len(output_masks)):
        for j in range(len(output_masks)):
            if i == j:
                continue
            if ((input_masks_[i] * input_masks_[j]).sum() > 0) and (input_pred_phrases[i].split("(")[0] != input_pred_phrases[j].split("(")[0]):
                # if two masks overlap and have different labels
                if float(input_pred_phrases[i].split("(")[1].split(")")[0]) < float(input_pred_phrases[j].split("(")[1].split(")")[0]):
                    # if the score of the first mask is lower than the second mask, remove overlapping area from the first mask
                    output_masks[i] = np.logical_and(output_masks[i], np.logical_not(input_masks_[j]))
                else:
                    # otherwise, remove overlapping area from the second mask
                    output_masks[j] = np.logical_and(output_masks[j], np.logical_not(input_masks_[i]))
    return output_masks.transpose(3, 0, 1, 2)[0]


groundingdino_model = None
sam_predictor = None
already_converted = {}
config_file = os.path.join(pretrained_models_folder, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")


def _find_higest_points(logits_map, num_top_points=20):
    if num_top_points == 0:
        return logits_map, []
    # find the highest points on the logits map
    gray = cv2.cvtColor(logits_map, cv2.COLOR_BGR2GRAY).astype("uint8")
    # find the highest points
    points = []
    for i in range(num_top_points):
        y, x = np.unravel_index(np.argmax(gray, axis=None), gray.shape)
        points.append((x, y))
        gray[y, x] = 0
    # draw points
    for point in points:
        cv2.drawMarker(logits_map, point, (0, 0, 255), cv2.MARKER_CROSS, 10, 3)
    return logits_map, points


def _find_contour_points(logits_map, num_points=5):
    if num_points == 0:
        return logits_map, []
    # find contours and get number of points on the contour, then draw the points on the image
    gray = cv2.cvtColor(logits_map, cv2.COLOR_BGR2GRAY).astype("uint8")
    ret, thresh = cv2.threshold(gray, 155, 255, 0)
    # erode to make the contour thinner
    kernel = np.ones((13, 13), np.uint8)
    # only apply erode when the image is large enough, otherwise, skip it
    if np.sum(thresh) > (gray.shape[0] * gray.shape[1] * 255 * 0.1):
        erode_iterations = int(np.log2(min(gray.shape[0], gray.shape[1])) - 1)
        thresh = cv2.erode(thresh, kernel, iterations=erode_iterations)

    # only use the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    points = []
    if len(largest_contour) > num_points:
        for i in range(0, len(largest_contour), len(largest_contour) // num_points):
            if len(points) == num_points:
                break
            x, y = largest_contour[i][0]
            points.append((x, y))

    # make sure the points are at the same number as num_points
    if len(points) == 0:
        raise ValueError("no points found")
    elif len(points) < num_points:
        for i in range(num_points - len(points)):
            points.append(points[-1])
    elif len(points) > num_points:
        points = points[:num_points]
    else:
        pass
    # draw points
    for point in points:
        # cv2.circle(logits_map, point, 3, (0, 0, 255), -1)
        cv2.drawMarker(logits_map, point, (0, 0, 255), cv2.MARKER_CROSS, 10, 3)

    return logits_map, points


def _process_logits(logits, pred_phrases, top_n_points):
    # print("logits", logits.shape)
    # torch.Size([3, 1, 468, 500])
    logits = logits.cpu().numpy()[:, 0, :, :]
    logits = ((logits - np.min(logits)) / (np.max(logits) - np.min(logits))) * 255
    logits_maps = []
    points_list = []
    for i, logits_map in enumerate(logits):
        try:
            logits_map = cv2.cvtColor(np.array(logits_map, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            logits_map, points = _find_higest_points(logits_map, num_top_points=top_n_points)
            if len(points) == 0:
                points = None
            cv2.putText(logits_map, pred_phrases[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            logits_maps.append(logits_map)
            points_list.append(points)
        except Exception as e:
            print("error in _process_logits", e)
            continue
    return logits_maps, points_list


def run_grounded_sam(
    input_image,
    text_prompt,
    box_threshold,
    text_threshold,
    iou_threshold,
    LABELS = [],
    IDS    = [],
    llm    = None,
    timer  = None,
    # for ablation study
    wo_post      = False,
    top_n_points = 20,
):
    global groundingdino_model, sam_predictor, already_converted

    # load image
    image_pil = input_image["image"].convert("RGB")
    transformed_image = transform_image(image_pil).to(device=device)
    size = image_pil.size

    if groundingdino_model is None:
        groundingdino_model = _load_model(config_file, groundingdino_ckpt, device=device)

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold)
    timer.check("get_grounding_output")

    # process boxes
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()

    # nms
    nms_idx      = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt   = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    if sam_predictor is None:
        # initialize SAM
        assert sam_ckpt, "sam_ckpt is not found!"
        sam = build_sam_vit_l(checkpoint=sam_ckpt)
        sam.to(device=device).eval()
        sam_predictor = SamPredictor(sam)
    sam_predictor.model.to(device=device)
    image = np.array(image_pil)
    sam_predictor.set_image(image)

    input_box = torch.tensor(boxes_filt, device=device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
    logits, _, _ = sam_predictor.predict_torch(
        point_coords     = None,
        point_labels     = None,
        boxes            = transformed_boxes,
        multimask_output = False,
        return_logits    = True,
        hq_token_only    = False,
    )
    timer.check("get prob")

    output_prob_maps, output_points = _process_logits(logits, pred_phrases, top_n_points=top_n_points)
    if top_n_points == 0:
        # processing without points prompt, for ablation study
        print("processing without points prompt, for ablation study")
        point_coords = None
        point_labels = None
    else:
        if None in output_points:
            point_coords = None
            point_labels = None
        else:
            point_coords = torch.tensor(np.array(output_points), device=device)
            point_coords = sam_predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
            point_labels = torch.ones(point_coords.shape[:2], device=device)
            # print("point_coords", point_coords.shape, point_labels.shape, transformed_boxes.shape)
            transformed_boxes = transformed_boxes[: point_coords.shape[0]]

    masks, _, _ = sam_predictor.predict_torch(
        point_coords     = point_coords,
        point_labels     = point_labels,
        boxes            = transformed_boxes,
        multimask_output = False,
        hq_token_only    = False,
    )
    masks = postprocess_masks(masks, pred_phrases)
    timer.check("postprocess_masks")

    label_image = Image.new("L", size, color=0)
    label_draw = np.array(label_image)
    output_labels = []
    for mask, pred_phrase in zip(masks, pred_phrases):
        try:
            label = pred_phrase.split("(")[0]
            if label in ["", " "]:
                # skip empty label
                continue
            elif label in LABELS:
                # no need to convert if it's one of the target labels
                post_label = label
            elif label in already_converted:
                # check if the label was converted before to save time and model calls
                post_label = already_converted[label]
                print("already converted: {} to {}".format(label, already_converted[label]))
            else:
                # convert the label using llm model
                label = label.replace(" ", "") if "-" in label else label
                if wo_post:
                    print("wo_post is True, for ablation study")
                    # skip post refinement, for ablation study
                    post_label = label
                else:
                    post_label = post_refinement(LABELS, label, llm=llm)
                print("convert from {} to {}".format(label, post_label))
                # add to the already_converted list, no matter it's in the list or not to save $!
                already_converted.update({label: post_label})
                if post_label not in LABELS:
                    raise ValueError("label not found, {} from {}".format(post_label, label))
            output_labels.append(post_label)
            label_index = LABELS.index(post_label)
            label_draw[mask] = IDS[label_index]
        except ValueError as e:
            print("e", e)
            print("label not found: ", pred_phrase)
            traceback.print_exc()
            continue
    timer.check("llm+draw label")
    return label_draw, boxes_filt, output_labels, output_prob_maps, output_points

import os
import cv2
import glob
import torch
import shutil
from termcolor import colored


import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from omegaconf import OmegaConf
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import multiprocessing as mp

from utils.labels_utils import Labels

# set up environment
from utils.env_utils import set_random_seed

set_random_seed(1024)


# save to excel
from openpyxl import Workbook
from openpyxl.styles import Font

# panda to excel
bold = Font(bold=True)
wb = Workbook()
sheet = wb["Sheet"]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    inters = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_inters = torch.histc(inters.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_inters
    return area_inters.cuda(), area_union.cuda(), area_target.cuda()


def compute_metrics(img_path):
    try:
        image_ori = cv2.imread(img_path)
        basename = img_path.replace(input_dir, "").replace("/", "_").split(".")[0][1:]
        label_gt_np = np.load(os.path.join(gt_dir, basename + ".npy"), allow_pickle=True)
        label_gt = torch.Tensor(label_gt_np).long().unsqueeze(0).cuda()
        npy_file_ours = os.path.join(ours_dir, basename + ".npy")
        if not os.path.exists(npy_file_ours):
            # if the file does not exist, assume the image is not annotated sucessfully, so the label is all zeros
            label_ours_np = np.zeros_like(label_gt_np)
        else:
            label_ours_np = np.load(npy_file_ours, allow_pickle=True)
        label_ours_np = cv2.resize(label_ours_np, (label_gt_np.shape[1], label_gt_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        label_ours = torch.Tensor(label_ours_np).long().unsqueeze(0).cuda()
        if use_baseline:
            npy_file_baseline = os.path.join(baseline_dir, basename + ".npy")
            if not os.path.exists(npy_file_baseline):
                # if the file does not exist, assume the image is not annotated sucessfully, so the label is all zeros
                label_base_np = np.zeros_like(label_gt_np)
            else:
                label_base_np = np.load(npy_file_baseline, allow_pickle=True)
            label_base = torch.Tensor(label_base_np).long().unsqueeze(0).cuda()
        else:
            label_base_np = np.zeros_like(label_ours_np)
            label_base = torch.zeros_like(label_ours)

        for id in L.IDS:
            if id == 0:
                continue
            label_gt[label_gt == id] = L.IDS.index(id)
            label_ours[label_ours == id] = L.IDS.index(id)
            if use_baseline:
                label_base[label_base == id] = L.IDS.index(id)

        intersection, union, target = intersectionAndUnionGPU(label_ours, label_gt, len(L.LABELS))
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        mIoU_ = sum(intersection) / sum(union + 1e-10)  # type: ignore
        # print("mIoU Ours:", mIoU_)
        if use_baseline:
            intersection1, union1, target1 = intersectionAndUnionGPU(label_base, label_gt, len(L.LABELS))
            intersection1, union1, target1 = intersection1.cpu().numpy(), union1.cpu().numpy(), target1.cpu().numpy()
            mIoU_1 = sum(intersection1) / sum(union1 + 1e-10)  # type: ignore
            # print("mIoU Base:", mIoU_1)

            if mIoU_ > mIoU_1:
                save_folder = output_dir_img
            else:
                save_folder = output_dir_img_ng
            os.makedirs(save_folder, exist_ok=True)
        else:
            intersection1, union1, target1 = 0, 0, 0
            save_folder = output_dir_img

        ours = L.draw_mask(label_ours_np, image_ori, print_label=False, only_label=only_label)
        gt = L.draw_mask(label_gt_np, image_ori, print_label=True, only_label=only_label)
        if use_baseline:
            base = L.draw_mask(label_base_np, image_ori, print_label=False, only_label=only_label)
            assert (
                base.shape == ours.shape and base.dtype == ours.dtype and base.shape == gt.shape and base.dtype == gt.dtype
            ), "base.shape != ours.shape or base.dtype != ours.dtype"
            res = cv2.hconcat([base, ours, gt])
        else:
            assert ours.shape == gt.shape and ours.dtype == gt.dtype, "ours.shape != gt.shape or ours.dtype != gt.dtype"
            res = cv2.hconcat([ours, gt])
            # exit()
        cv2.imwrite(os.path.join(save_folder, basename + ".jpg"), res, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return intersection, union, target, intersection1, union1, target1
    except Exception as e: # noqa
        print(f"Failed to compute metrics for {img_path}, returning 0 and skip")
        # traceback.print_exc()
        return 0, 0, 0, 0, 0, 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, help="path to config file")
    parser.add_argument("--ours", default=None, required=True, help="")
    parser.add_argument("--baseline", default=None, required=False, help="if necessary")
    parser.add_argument("--only_label", default=False, action="store_true", help="only show label")
    parser.add_argument("--debug", default=False, action="store_true", help="stop multiprocessing for debugging")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # init labels and gpt prompt
    L = Labels(config=config)
    only_label = args.only_label

    # get input images
    input_dir = config.input_dir
    all_imgs = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

    if args.debug:
        all_imgs = natsorted(all_imgs)[: len(all_imgs) // 20]
        print(colored("Using only 5% of the images", "red"))
    else:
        all_imgs = natsorted(all_imgs)
        print(colored("Using all images", "green"))
    print("Total numbers of images:", len(all_imgs))

    assert hasattr(config, "gt_dir"), "config file must have gt_dir to evaluate"
    gt_dir       = config.gt_dir
    ours_dir     = args.ours
    baseline_dir = args.baseline
    use_baseline = True if baseline_dir is not None else False
    output_dir   = os.path.join(os.path.dirname(__file__), "results", config.Name)
    shutil.rmtree(output_dir, ignore_errors=True) if os.path.exists(output_dir) else None
    output_dir_img    = os.path.join(output_dir, "images")
    output_dir_img_ng = os.path.join(output_dir, "images_ng")
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_img_ng, exist_ok=True)

    union_meter, target_meter, intersection_meter = AverageMeter(), AverageMeter(), AverageMeter()
    union_meter1, target_meter1, intersection_meter1 = AverageMeter(), AverageMeter(), AverageMeter()

    if args.debug:
        intersection, union, target, intersection1, union1, target1 = [], [], [], [], [], []
        for img_path in tqdm(all_imgs):
            res = compute_metrics(img_path)
            intersection.append(res[0])
            union.append(res[1])
            target.append(res[2])
            intersection1.append(res[3])
            union1.append(res[4])
            target1.append(res[5])
    else:
        # use half of the cores, max 16
        pool_size = min(mp.cpu_count() // 2, 32)
        with mp.Pool(processes=pool_size) as p:
            results = list(tqdm(p.imap(compute_metrics, all_imgs), total=len(all_imgs)))
        intersection, union, target, intersection1, union1, target1 = zip(*results)

    for i in tqdm(range(len(all_imgs))):
        intersection_meter.update(intersection[i])
        union_meter.update(union[i])
        target_meter.update(target[i])
        intersection_meter1.update(intersection1[i])
        union_meter1.update(union1[i])
        target_meter1.update(target1[i])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)  # type: ignore
    print("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(mIoU, mAcc, allAcc))
    sheet.append(["class", "name", "mIoU", "mAcc"])
    #   , "allAcc"])
    sheet.append(["all", "ours", mIoU, mAcc])
    # , allAcc])

    if use_baseline:
        iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
        accuracy_class1 = intersection_meter1.sum / (target_meter1.sum + 1e-10)
        mIoU1 = np.mean(iou_class1)
        mAcc1 = np.mean(accuracy_class1)
        allAcc1 = sum(intersection_meter1.sum) / (sum(target_meter1.sum) + 1e-10)  # type: ignore
        print("Val result: mIoU1/mAcc1/allAcc1 {:.4f}/{:.4f}/{:.4f}.".format(mIoU1, mAcc1, allAcc1))
        sheet.append(["", "baseline", mIoU1, mAcc1])
        #   , allAcc1])

        if mIoU > mIoU1:  # type: ignore
            sheet["C2"].font = bold
        else:
            sheet["C3"].font = bold
        if mAcc > mAcc1:  # type: ignore
            sheet["D2"].font = bold
        else:
            sheet["D3"].font = bold
        if allAcc > allAcc1:
            sheet["E2"].font = bold
        else:
            sheet["E3"].font = bold
        data = {"class": ["all"], "ours": [mIoU], "baseline": [mIoU1]}
        data1 = {"class": ["all"], "ours": [mAcc], "baseline": [mAcc1]}
    else:
        data = {"class": ["all"], "ours": [mIoU]}
        data1 = {"class": ["all"], "ours": [mAcc]}

    for i in range(len(L.LABELS)):
        iou = iou_class[i]  # type: ignore
        acc = accuracy_class[i]  # type: ignore
        # print("{} {} iou/accuracy: {:.4f}/{:.4f}.".format(i, L.LABELS[i], iou, acc))
        sheet.append([L.LABELS[i], "ours", iou, acc, ""])
        data["class"].append(L.LABELS[i])
        data["ours"].append(iou)
        data1["class"].append(L.LABELS[i])
        data1["ours"].append(acc)

        if use_baseline:
            iou1 = iou_class1[i]
            acc1 = accuracy_class1[i]
            # print("{} {} iou1/accuracy1: {:.4f}/{:.4f}.".format(i, L.LABELS[i], iou1, acc1))
            sheet.append(["", "baseline", iou1, acc1, ""])
            data["baseline"].append(iou1)
            data1["baseline"].append(acc1)

            if iou > iou1:  # type: ignore
                sheet["C" + str(2 * i + 4)].font = bold
            else:
                sheet["C" + str(2 * i + 5)].font = bold
            if acc > acc1:  # type: ignore
                sheet["D" + str(2 * i + 4)].font = bold
            else:
                sheet["D" + str(2 * i + 5)].font = bold

    wb.save(os.path.join(output_dir, "result.xlsx"))
    # get only mIoU, plot bar, make baseline orange, ours blue
    pd.DataFrame(data).plot.bar(x="class", rot=0, color=["blue", "orange"], title="mIoU", ylabel="mIoU", xlabel="class", figsize=(20, 10))
    plt.savefig(os.path.join(output_dir, "mIoU.png"))
    pd.DataFrame(data1).plot.bar(x="class", rot=0, color=["blue", "orange"], title="mAcc", ylabel="mAcc", xlabel="class", figsize=(20, 10))
    plt.savefig(os.path.join(output_dir, "mAcc.png"))

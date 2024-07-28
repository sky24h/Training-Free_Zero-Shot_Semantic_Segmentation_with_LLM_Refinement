import cv2
import numpy as np

COCO_CATEGORIES = [
    # borrowed from detectron2
    # https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_coco_stuff_10k.py
    {"color": [0, 0, 0], "isthing": 0, "id": 0, "name": "unlabeled"},
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
]


def create_coco_colormap(IDs):
    all_colors = []
    vis_colors = [category["color"] for category in COCO_CATEGORIES]
    used_ids = [category["id"] for category in COCO_CATEGORIES]
    all_colors = [vis_colors[used_ids.index(id)] if id in used_ids else [0, 0, 0] for id in range(max(IDs)+1)]
    return np.array(all_colors, dtype=int)


def create_cityscapes_colormap(IDs):
    vis_colors = [
        (0, 0, 0),
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ]

    all_colors = [vis_colors[IDs.index(id)] if id in IDs else [0, 0, 0] for id in range(max(IDs)+1)]
    return np.array(all_colors, dtype=int)

def create_pascal_label_colormap(n_labels=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_labels, 3), dtype=np.uint8)
    for i in range(n_labels):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap



class Labels:
    def __init__(self, config=None):
        max_label_num = 200
        if config is not None:
            self.LABELS = config.label_list.split(", ")
            self.IDS = config.mask_ids if hasattr(config, "mask_ids") else [i for i in range(len(self.LABELS))]
            print("self.IDS", self.IDS)
            if len(self.LABELS) > max_label_num:
                raise ValueError(f"Too many labels! The maximum number of labels is {max_label_num}.")
        else:
            raise NotImplementedError("config is None")

        if "COCO" in config.Name:
            self.COLORS = create_coco_colormap(self.IDS)
        elif "City" in config.Name:
            self.COLORS = create_cityscapes_colormap(self.IDS)
        else:
            # default to pascal label colormap
            self.COLORS = create_pascal_label_colormap()

        assert len(self.COLORS) >= len(self.LABELS), f"len(self.COLORS)={len(self.COLORS)} < len(self.LABELS)={len(self.LABELS)}"

    def check_labels(self, labels_list):
        output_labels_list = []
        for labels in labels_list:
            output_labels = []
            labels = labels.split(", ")
            for label in labels:
                if label == "background":
                    # skip the background label
                    continue
                if label in self.LABELS:
                    output_labels.append(label)
            output_labels = list(set(output_labels))
            output_labels_list.append(", ".join(output_labels))
        return output_labels_list

    def draw_mask(self, label_ori, image_ori, print_label=False, tag="", only_label=False):
        label_ori = label_ori.astype(np.uint8)
        label = np.zeros_like(image_ori, dtype=np.uint8)
        # print("{}: {}".format(tag, np.unique(label_ori)))
        for id in np.unique(label_ori):
            # print("id", id)
            if id == 0 or id == 255:
                continue
            elif id not in self.IDS:
                print(f"Label {id} is not in the label list.")
                continue
            i = self.IDS.index(id)
            center = np.mean(np.argwhere(label_ori == id), axis=0).astype(np.int64)
            label[label_ori == id] = self.COLORS[id]
            if print_label:
                # add text in the center of the mask
                cv2.putText(label, self.LABELS[i], (center[1], center[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # print(i, self.LABELS[i])
        # RGB to BGR
        label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(label, 0.6, image_ori, 0.4, 0) if not only_label else label

    def find_gt_labels(self, label_gt):
        label_gt = label_gt.astype(np.uint8)
        label_gt_list = []
        for id in np.unique(label_gt):
            if id == 0 or id == 255:
                continue
            elif id not in self.IDS:
                print(f"Label {id} is not in the label list.")
                continue
            i = self.IDS.index(id)
            label_gt_list.append(self.LABELS[i])
        return label_gt_list

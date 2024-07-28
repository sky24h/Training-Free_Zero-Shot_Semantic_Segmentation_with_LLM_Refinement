# CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/VOC2012.yaml --reset --draw_bbox --debug
CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/DRAM.yaml --reset --draw_bbox --debug
CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/Cityscapes.yaml --reset --draw_bbox --debug
CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/COCO-81.yaml --reset --draw_bbox --debug
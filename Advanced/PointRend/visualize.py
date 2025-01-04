import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import random
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.projects import point_rend
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# 配置模型
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file("./config/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.RESNETS.DEPTH = 50
cfg.DATASETS.TRAIN = ("ade20k_sem_seg_train",)
cfg.DATASETS.TEST = ("ade20k_sem_seg_val",)
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.SOLVER.IMS_PER_BATCH = 24
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 8000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 150  # ADE20k有150个类别
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 150
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 150

cfg.INPUT.MIN_SIZE_TRAIN = (512,)
cfg.INPUT.MIN_SIZE_TEST = (512,)
cfg.INPUT.MAX_SIZE_TRAIN = 512
cfg.INPUT.MAX_SIZE_TEST = 512
cfg.INPUT.CROP.ENABLED = False

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("ade20k_sem_seg_val")
metadata = MetadataCatalog.get("ade20k_sem_seg_val")

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    outputs = outputs["sem_seg"].argmax(dim=0)
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
    v = v.draw_sem_seg(outputs.to("cpu"))
    v = v.draw_dataset_dict(d)
    
    cv2.imshow("Result", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.imwrite("sample_result.jpg", v.get_image()[:, :, ::-1])
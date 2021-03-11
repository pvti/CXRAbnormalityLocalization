import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

from config import thing_classes, category_name_to_id

import cv2
import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.config.config import CfgNode as CN
from tqdm import tqdm

from prepare_data import get_vinbigdata_dicts
from flag import Flags
from my_trainer import MyTrainer
from utils import save_yaml

setup_logger()

def verify_load_data(dataset_dicts, num_samples, metadata):
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite("samples/"+os.path.basename(d["file_name"]), vis.get_image()[:, :, ::-1]) 
    return 0

def train_net(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer

if __name__ == "__main__":
    flags_dict = {
        "debug": False,
        "outdir": "results/v9", 
        "imgdir_name": "vinbigdata-chest-xray-resized-png-256x256",
        "split_mode": "valid20",
        "iter": 10000,
        "roi_batch_size_per_image": 512,
        "eval_period": 1000,
        "lr_scheduler_name": "WarmupCosineLR",
        "base_lr": 0.001,
        "num_workers": 4,
        "aug_kwargs": {
            "HorizontalFlip": {"p": 0.5},
            "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
            "RandomBrightnessContrast": {"p": 0.5}
        }
    }

    flags = Flags().update(flags_dict)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)
    flags_dict = dataclasses.asdict(flags)
    save_yaml(outdir / "flags.yaml", flags_dict)

    inputdir = Path("./")
    datadir = inputdir / "vinbigdata-chest-xray-abnormalities-detection"
    imgdir = inputdir / flags.imgdir_name

    train_df = pd.read_csv(datadir / "train.csv")
    train = train_df

    train_data_type = flags.train_data_type
    if flags.use_class14:
        thing_classes.append("No finding")
    
    split_mode = flags.split_mode
    if split_mode == "all_train":
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
    elif split_mode == "valid20":
        # To get number of data...
        n_dataset = len(
            get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            )
        )
        n_train = int(n_dataset * 0.8)
        print("n_dataset", n_dataset, "n_train", n_train)
        rs = np.random.RandomState(flags.seed)
        inds = rs.permutation(n_dataset)
        train_inds, valid_inds = inds[:n_train], inds[n_train:]
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=train_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
        DatasetCatalog.register(
            "vinbigdata_valid",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=valid_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_valid").set(thing_classes=thing_classes)
    else:
        raise ValueError(f"[ERROR] Unexpected value split_mode={split_mode}")

    dataset_dicts = get_vinbigdata_dicts(imgdir, train, debug=debug)
    
    # Visualize data...
    vinbigdata_metadata = MetadataCatalog.get("vinbigdata_train")
    
    #verify_load_data(dataset_dicts, 10, vinbigdata_metadata)    
    
    cfg = get_cfg()
    cfg.aug_kwargs = CN(flags.aug_kwargs)

    cfg.OUTPUT_DIR = str(outdir)
    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = ("vinbigdata_train",)
    if split_mode == "all_train":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ("vinbigdata_valid",)
        cfg.TEST.EVAL_PERIOD = flags.eval_period

    cfg.DATALOADER.NUM_WORKERS = flags.num_workers
# Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.SOLVER.IMS_PER_BATCH = flags.ims_per_batch
    cfg.SOLVER.LR_SCHEDULER_NAME = flags.lr_scheduler_name
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000  # Small value=Frequent save need a lot of storage.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

    trainer = train_net(cfg)

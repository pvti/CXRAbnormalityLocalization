import os
import pandas as pd
from utils import load_yaml
from flag import Flags
from pathlib import Path
from tqdm import tqdm
from math import ceil
from typing import Any, Dict, List
import cv2

from config import thing_classes, category_name_to_id
from prepare_data import get_vinbigdata_dicts_test
from prediction import format_pred, predict_batch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

if __name__ == '__main__':
    inputdir = Path("./")
    traineddir = inputdir / "results/v9"
    flags: Flags = Flags().update(load_yaml(str(traineddir / "flags.yaml")))
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)
    
    # --- Read data ---
    datadir = inputdir / "vinbigdata-chest-xray-abnormalities-detection"
    if flags.imgdir_name == "vinbigdata-chest-xray-resized-png-512x512":
        imgdir = inputdir/ "vinbigdata"
    else:
        imgdir = inputdir / flags.imgdir_name
    
    # Read in the data CSV files
    test_meta = pd.read_csv(inputdir / "vinbigdata-testmeta" / "test_meta.csv")
    sample_submission = pd.read_csv(datadir / "sample_submission.csv")
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(outdir)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("vinbigdata_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    ### --- Inference & Evaluation ---
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = str(traineddir/"model_final.pth")
    print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set a custom testing threshold
    print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    predictor = DefaultPredictor(cfg)
    
    DatasetCatalog.register(
        "vinbigdata_test", lambda: get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug)
    )
    MetadataCatalog.get("vinbigdata_test").set(thing_classes=thing_classes)
    metadata = MetadataCatalog.get("vinbigdata_test")
    dataset_dicts = get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug)
    
    if debug:
        dataset_dicts = dataset_dicts[:100]
    
    results_list = []
    index = 0
    batch_size = 4
    
    for i in tqdm(range(ceil(len(dataset_dicts) / batch_size))):
        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dicts))))
        dataset_dicts_batch = [dataset_dicts[i] for i in inds]
        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        outputs_list = predict_batch(predictor, im_list)
    
        for im, outputs, d in zip(im_list, outputs_list, dataset_dicts_batch):
            resized_height, resized_width, ch = im.shape
            # outputs = predictor(im)
            if index < 5:
                # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # cv2_imshow(out.get_image()[:, :, ::-1])
                cv2.imwrite(str(outdir / f"pred_{index}.jpg"), out.get_image()[:, :, ::-1])
    
            image_id, dim0, dim1 = test_meta.iloc[index].values
    
            instances = outputs["instances"]
            if len(instances) == 0:
                # No finding, let's set 14 1 0 0 1 1x.
                result = {"image_id": image_id, "PredictionString": "14 1.0 0 0 1 1"}
            else:
                # Find some bbox...
                # print(f"index={index}, find {len(instances)} bbox.")
                fields: Dict[str, Any] = instances.get_fields()
                pred_classes = fields["pred_classes"]  # (n_boxes,)
                pred_scores = fields["scores"]
                # shape (n_boxes, 4). (xmin, ymin, xmax, ymax)
                pred_boxes = fields["pred_boxes"].tensor
    
                h_ratio = dim0 / resized_height
                w_ratio = dim1 / resized_width
                pred_boxes[:, [0, 2]] *= w_ratio
                pred_boxes[:, [1, 3]] *= h_ratio
    
                pred_classes_array = pred_classes.cpu().numpy()
                pred_boxes_array = pred_boxes.cpu().numpy()
                pred_scores_array = pred_scores.cpu().numpy()
    
                result = {
                    "image_id": image_id,
                    "PredictionString": format_pred(
                        pred_classes_array, pred_boxes_array, pred_scores_array
                    ),
                }
            results_list.append(result)
            index += 1
    submission_det = pd.DataFrame(results_list, columns=['image_id', 'PredictionString'])
    submission_det.to_csv(outdir/"submission.csv", index=False)

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from detectron2.structures import BoxMode
from tqdm import tqdm

def get_vinbigdata_dicts(
        imgdir: Path,
        train_df: pd.DataFrame,
        train_data_type: str = "original",
        use_cache: bool = True,
        debug: bool = True,
        target_indices: Optional[np.ndarray] = None,
        use_class14: bool = False,
        ):
    debug_str = f"_debug{int(debug)}"
    train_data_type_str = f"_{train_data_type}"
    class14_str = f"_14class{int(use_class14)}"
    cache_path = Path(".") / f"dataset_dicts_cache{train_data_type_str}{class14_str}{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        print(imgdir / "train_meta.csv")
        train_meta = pd.read_csv(imgdir / "train_meta.csv")
        if debug:
            train_meta = train_meta.iloc[:500] #for debug

        #load 1 image to get image size
        image_id = train_meta.loc[0, "image_id"]
        image_path = str(imgdir / "train" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, train_meta_row in tqdm(train_meta.iterrows(), total=len(train_meta)):
            record = {}

            image_id, height, width = train_meta_row.values
            filename = str(imgdir / "train" / f"{image_id}.png")
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            objs = []
            for index2, row in train_df.query("image_id == @image_id").iterrows():
                class_id = row["class_id"]
                if class_id == 14:
                    #It is "No finding"
                    if use_class14:
                        #use this No finding class with the bbox covering all image area
                        bbox_resized = [0, 0, resized_width, resized_height]
                        obj = {
                                "bbox":bbox_resized,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": class_id,
                        }
                        objs.append(obj)
                    else:
                        #this annotaor does find any label, skip
                        pass
                else:
                    h_ratio = resized_height / height
                    w_ratio = resized_width / width
                    bbox_resized = [
                            float(row["x_min"]) * w_ratio,
                            float(row["y_min"]) * h_ratio,
                            float(row["x_max"]) * w_ratio,
                            float(row["y_max"]) * h_ratio,
                    ]
                    obj = {
                            "bbox": bbox_resized,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": class_id,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode = "wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]
    return dataset_dicts

def get_vinbigdata_dicts_test(
        imgdir: Path,
        test_meta: pd.DataFrame,
        use_cache: bool = True,
        debug: bool = True,
        ):
    debug_str = f"_debug{int(debug)}"
    cache_path = Path(".") / f"dataset_dicts_cache_test{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        if debug:
            test_meta = test_meta.iloc[:500] #for debug

        #load 1 image to get image size
        image_id = test_meta.loc[0, "image_id"]
        image_path = str(imgdir / "test" /f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width = test_meta_row.values
            filename = str(imgdir / "test" / f"{image_id}.png")
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width

            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts

# --- configs ---
thing_classes = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]
category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}

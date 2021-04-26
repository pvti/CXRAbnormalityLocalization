import numpy as np
import pandas as pd
from glob import glob
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import seaborn as sns
import argparse
from os import listdir
from os.path import isfile, join
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description="Train yolov5")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/tienpv/CXRDet/data/vinbigdata-512-image-dataset/vinbigdata/train.csv",
        help="Path to train meta data cvs file",
    )
    return parser

def preprocess(train_df):
    train_df['image_path'] = '/home/tienpv/CXRDet/data/vinbigdata-512-image-dataset/vinbigdata/train/'+train_df.image_id+'.png'
    #keep 14 classes only
    train_df = train_df[train_df.class_id!=14].reset_index(drop = True)
    
    #covert to relative coordiantes
    train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
    train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis =1)
    
    train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
    train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis =1)
    
    train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
    train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)
    
    train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
    train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)
    
    train_df['area'] = train_df['w']*train_df['h']
    
    return train_df

def split(train_df):
    gkf  = GroupKFold(n_splits = 1)
    train_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
        train_df.loc[val_idx, 'fold'] = fold
    train_df.head()
    
    train_files = []
    val_files   = []
    val_files += list(train_df[train_df.fold==fold].image_path.unique())
    train_files += list(train_df[train_df.fold!=fold].image_path.unique())
    
    return train_df, train_files, val_files

def prepare_yolo_format(train_files, val_files):
    os.makedirs('/home/tienpv/CXRDet/data/vinbigdata/labels/train', exist_ok = True)
    os.makedirs('/home/tienpv/CXRDet/data/vinbigdata/labels/val', exist_ok = True)
    os.makedirs('/home/tienpv/CXRDet/data/vinbigdata/images/train', exist_ok = True)
    os.makedirs('/home/tienpv/CXRDet/data/vinbigdata/images/val', exist_ok = True)
    label_dir = '/home/tienpv/CXRDet/data/vinbigdata-yolo-labels-dataset/labels'
    for file in tqdm(train_files):
        shutil.copy(file, '/home/tienpv/CXRDet/data/vinbigdata/images/train')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), '/home/tienpv/CXRDet/data/vinbigdata/labels/train')
        
    for file in tqdm(val_files):
        shutil.copy(file, '/home/tienpv/CXRDet/data/vinbigdata/images/val')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), '/home/tienpv/CXRDet/data/vinbigdata/labels/val')
    
if __name__ == "__main__":
    args = get_parser().parse_args()
    train_df = pd.read_csv(args.input)
    train_df = preprocess(train_df)
    train_df, train_files, val_files = split(train_df)
    prepare_yolo_format(train_files, val_files)
    class_ids, class_names = list(zip(*set(zip(train_df.class_id, train_df.class_name))))
    classes = list(np.array(class_names)[np.argsort(class_ids)])
    classes = list(map(lambda x: str(x), classes))

    cwd = './'
    with open(join( cwd , 'train.txt'), 'w') as f:
        for path in glob('/home/tienpv/CXRDet/data/vinbigdata/images/train/*'):
            f.write(path+'\n')
            
    with open(join( cwd , 'val.txt'), 'w') as f:
        for path in glob('/home/tienpv/CXRDet/data/vinbigdata/images/val/*'):
            f.write(path+'\n')
    
    data = dict(
        train =  join( cwd , 'train.txt') ,
        val   =  join( cwd , 'val.txt' ),
        nc    = 14,
        names = classes
        )
    
    with open(join( cwd , 'vinbigdata.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

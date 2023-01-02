# CXRAbnormalityLocalization [![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fpvtien96%2FCXRAbnormalityLocalization&countColor=%232ccce4)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fpvtien96%2FCXRAbnormalityLocalization)
> [**Chest X-ray abnormalities localization using convolutional neural network**](https://drive.google.com/drive/folders/1yylqLDt59O3vxGkRRzVQaaS0zut6Cdg9?usp=sharing),            
> Van-Tien Pham, Minh-Cong Tran, Stanley Zheng, Tri-Minh Vu, Shantanu Nath;        

![](readme/proposedFramework.png)

> Contact: [pvtien96@gmail.com](mailto:pvtien96@gmail.com). Discussions are welcome!

## Abstract
Convolutional neural network algorithms have been applied widely in chest X-ray interpretation thanks to the availability of high-quality datasets. Among them, VinDr-CXR is one of the latest public dataset including 18000 expert-annotated images labeled into 22 local position-specific abnormalities and 6 global suspected diseases. A proposed deep learning algorithms based on Faster-RCNN, Yolov5 and EfficientDet frameworks were developed and investigated in the task of multi-class clinical detection from chest radiography. The ground truth was defined by radiologist-adjudicated image review. Their performance was evaluated by the mean average precision (mAP 0.4), which can be accessed via Kaggle's server. The results shows the best performance belonging to ensembled detector model combined with EfficientNet as the classifier with the accuracy peak of 0.292. As a trade-off, ensembling detectors was much slower, which increases computing time by 3.75, 5 and 2.25 times compared to FasterRCNN, Yolov5 and EfficientDet, respectively. Overall, the classifiers shows constantly improvement on all detector models, which is highly recommended for further research. All of this aspects should be considered to address the real-world CXR diagnosis where the accuracy and computing cost are the most concerned.

## News
- **[2021.10.15]** [Video presentation](https://www.youtube.com/watch?v=bJv3Nza3JuM)
[![https://www.youtube.com/watch?v=bJv3Nza3JuM](https://user-images.githubusercontent.com/25927039/139654886-a13887ac-b59c-4423-988d-4b6c78cea1af.png)](https://www.youtube.com/watch?v=bJv3Nza3JuM)
- **[2021.08.13]** Paper is accepted at the [2021 International Conference on Advanced Technologies for Communications (ATC)](https://atc-conf.org/).
- **[2021.03.30]** Finnish in top 10%.
- **[2021.03.01]** Build team and join [VinDr-CXR Kaggle competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection).


## Main results

Evaluation of the proposed framework on VinDr-CXR test dataset.

|   Detector   |              | Accuracy (mAP@0.4) |                |       |         Performance         |                      |
|:------------:|:------------:|:------------------:|----------------|:-----:|:---------------------------:|:--------------------:|
|              | Single model |      Resnet50      | EficientNet-B7 | Speed | GPU memory requirement (MB) | Training time (hour) |
| YOLOv5       |     0.21     |        0.246       |      0.269     |   15  |             3291            |           7          |
| FasterRCNN   |     0.248    |        0.263       |      0.278     |   20  |             2076            |          9.5         |
| EfficientDet |     0.269    |        0.28        |      0.273     |   9   |             3685            |          12          |
| Ensemble     |     0.272    |        0.285       |      0.292     |   4   |             3685            |         30.5         |


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Model zoo

Trained models are available in the [MODEL_ZOO.md](readme/MODEL_ZOO.md).

## Dataset zoo

Please see [DATASET_ZOO.md](readme/DATASET_ZOO.md) for detailed description of the training/evaluation datasets.

## Getting Started

Follow the aforementioned instructions to install environments and download models and datasets.

[GETTING_STARTED.md](readme/GETTING_STARTED.md) provides a brief intro of the usage of builtin command-line tools.

## License

Code is released under the [Apache 2.0 license](LICENSE).

## Citing

If you use this work in your research or wish to refer to the results, please use the following BibTeX entry.

```BibTeX
@INPROCEEDINGS{Pham2110:Chest,
AUTHOR="Van-Tien Pham and Cong-Minh Tran and Stanley Zheng and Tri-Minh Vu and
Shantanu Nath",
TITLE="Chest X-ray abnormalities localization via ensemble of deep convolutional
neural networks",
BOOKTITLE="2021 International Conference on Advanced Technologies for Communications
(ATC) (ATC'21)",
ADDRESS="Ho Chi Minh City, Vietnam",
DAYS=13,
MONTH=oct,
YEAR=2021,
KEYWORDS="chest x-ray; chest radiograph; localization; classification; thoracic
abnormalities; VinDrCXR",
ABSTRACT="Convolutional neural network algorithms have been applied widely in chest
X-ray interpretation thanks to the availability of high-quality datasets.
Among them, VinDr-CXR is one of the latest public dataset including 18000
expert-annotated images labeled into 22 local position-specific
abnormalities and 6 global suspected diseases. A proposed deep learning
algorithms based on Faster-RCNN, Yolov5 and EfficientDet frameworks were
developed and investigated in the task of multi-class clinical detection
from chest radiography. The ground truth was defined by
radiologist-adjudicated image review. Their performance was evaluated by
the mean average precision (mAP 0.4), which can be accessed via Kaggle's
server. The results shows the best performance belonging to ensembled
detector model combined with EfficientNet as the classifier with the
accuracy peak of 0.292. As a trade-off, ensembling detectors was much
slower, which increases computing time by 3.75, 5 and 2.25 times compared
to FasterRCNN, Yolov5 and EfficientDet, respectively. Overall, the
classifiers shows constantly improvement on all detector models, which is
highly recommended for further research. All of this aspects should be
considered to address the real-world CXR diagnosis where the accuracy and
computing cost are the most concerned."
}
```

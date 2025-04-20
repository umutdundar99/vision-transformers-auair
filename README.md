# transformer-detection
 You Only Look at One Sequence: Object detection on AU-AIR dataset

#  Introduction
This repository contains the implementation of You Only Look at One Sequence on the AU-AIR dataset. The paper proposes a novel transformer-based object detection framework that can effectively detect objects in a single sequence of images. The framework is designed to work with the AU-AIR dataset, which contains a large number of annotated sequences for training and evaluation.
The YOLOS-tiny architecture was trained from scratch using an input resolution of 512×864, rather than leveraging pre-trained weights, as the available pre-trained models were trained on the ImageNet-1K dataset with a resolution of 800×1333. Due to limited computational resources, we opted for training at a lower resolution without pre-training.

# Dataset
The AU-AIR dataset is a large-scale dataset for object detection in aerial images. The dataset contains a large number of annotated sequences, which can be used for training and evaluation of object detection models. The dataset can be downloaded from the official website: [AU-AIR Dataset](https://bozcani.github.io/auairdataset/).
The dataset is split for training, validation, and testing with %75, %15 and %10 respectively. Please locate the dataset in the `dataset` folder with the  "auair2019" folder name.

# Key Features 
- **Model**: YOLOS-Tiny implemented with PyTorch Lightning
- **Training from Scratch**: Random weight initialization without any pre-trained weights
- **Dataset**: The images are resized to 512x864 and the bounding boxes are resized accordingly.
- **Batch Size**: 30 (10 batch size and 3 accumulate gradient steps due to memory constraints)
- **Hungarian Matching**: For unique box-to-ground truth assignment
- **Loss Functions**:
  Cross-Entropy for classification
  L1 and GIoU for bounding boxes
- **Augmentations** (Albumentations):
   Random brightness/contrast/gamma
  Safe random cropping, rotation, and flipping
  Bounding box aware (`BboxParams`)
- **Evaluation**:
 COCO mAP @ IoU=0.5 and [0.5:0.95]
 TorchMetrics compatible
- **No post-filtering** (as in YOLOS paper)

## Installation

Please create an environment, preferably a virtual environment.

### Create a Virtual Environment
```bash
python -m venv .venv
```
### Activate the Environment

#### Windows

```bash
source venv\Scripts\activate
```

#### Ubuntu
```bash
source venv/bin/activate
```

### Download the dependencies
```bash
pip install -e .
```

## Usage

```bash
python3.10 -m vision_transformers_auair
```
After that, you will be able to see the training process in the terminal. Please configure WandbLogger for your own account.

**Please note that this code trains YOLOS-tiny from scratch. If you want to use pre-trained weights, please
download the weights from [YOLOS](https://github.com/hustvl/YOLOS) and do the necessary changes in the code since the number of classes is different. The code is not tested with pre-trained weights because the purpose of this project is to train the model from scratch** 

# Results
Since there were not enough resources to train the YOLOS architecture on the AU-AIR dataset long enough, the final results will be shared later.

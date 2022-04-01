# CMIC-Retrieval
Code for **Single Image 3D Shape Retrieval via Cross-Modal Instance and Category Contrastive Learning**.  **ICCV 2021.**

![Overview](/images/teaser.png)



## Introduction

In this work, we tackle the problem of single image-based 3D shape retrieval (IBSR), where we seek to find the most matched shape of a given single 2D image from a shape repository. Most of the existing works  learn to embed 2D images and 3D shapes into a common feature space and perform metric learning using a triplet loss. Inspired by the great success in recent contrastive learning works on self-supervised  representation learning, we propose a novel IBSR pipeline leveraging contrastive learning. We note that adopting such cross-modal contrastive learning between 2D images and 3D shapes into IBSR tasks is non-trivial and challenging: contrastive learning requires very strong data augmentation in constructed positive pairs to learn the feature invariance, whereas traditional metric learning works do not have this requirement. However, object shape and appearance are entangled in 2D query images, thus making the learning task more difficult than contrasting single-modal data. To mitigate the challenges, we propose to use multi-view grayscale rendered images from the 3D shapes as a shape representation. We then introduce a strong data augmentation technique based on color transfer, which can significantly but naturally change the appearance of the query image, effectively satisfying the need for contrastive learning. Finally, we propose to incorporate a novel category-level contrastive loss that helps distinguish similar objects from different categories, in addition to classic instance-level contrastive loss. Our experiments demonstrate that our approach achieves the best performance on all the three popular IBSR benchmarks, including Pix3D, Stanford Cars, and Comp Cars, outperforming the previous state-of-the-art from 4% - 15% on retrieval accuracy.



## About this repository

This repository provides **data**, **pre-trained models** and **code**.



## Installation
```zsh
# create anoconda environment
## please make sure that python version >= 3.7 (required by jittor)
conda create -n ibsr_jittor python=3.7
conda activate ibsr_jittor

# jittor installation
python3.7 -m pip install jittor
## testing jittor
### if errors appear, you can follow the instructions of jittor to fix them.
python3.7 -m jittor.test.test_example
# testing for cudnn
python3.7 -m jittor.test.test_cudnn_op

# other pickages
pip install pyyaml
pip install scikit-learn
pip install matplotlib
pip install scikit-image
pip install argparse
```



## How to use

```zsh
# download pre-trained models, data and official ResNet pre-trained models from this links:
https://1drv.ms/u/s!Ams-YJGtFnP7mTQOACYHco1s2gXE?e=c87UnV

# put the unzip folder pre_trained, pretrained_resnet, data under IBSR_jittor/code
cd IBSR_jittor/code

# all codes are test under a single Nvidia RTX3090, Ubuntu 20.04
# training
python RetrievalNet.py --config ./configs/pix3d.yaml

# testing
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode simple
# for full test
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode full
# for shapenet test
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode shapenet

# pay attention to:
# model_std_bin128 and model_std_ptc10k_npy are not uploaded.
# For model_std_ptc10k_npy, we randomly sample 10k points from the mesh by python igl package.
# For model_std_bin128, please refer to https://www.patrickmin.com/viewvox/ for more information.
```




## Citations
```
@inProceedings{lin2021cmic,
	title={Single Image 3D Shape Retrieval via Cross-Modal Instance and Category Contrastive Learning},
	author={Lin, Ming-Xian and Yang, Jie and Wang, He and Lai, Yu-Kun and Jia, Rongfei and Zhao, Binqiang and Gao, Lin},
	year={2021},
	booktitle={International Conference on Computer Vision (ICCV)}
}
```



## Updates
- [Apr 1, 2021] Pre-trained Models, Data and revised Code released. 
- [Oct 1, 2021] Preliminary version of Data and Code released. For more code and data, coming soon. Please follow our updates.


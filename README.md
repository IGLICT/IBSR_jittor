# CMIC-Retrieval
Code for **Single Image 3D Shape Retrieval via Cross-Modal Instance and Category Contrastive Learning**.  **ICCV 2021.**

![Overview](/images/teaser.png)



## Introduction

In this work, we tackle the problem of single image-based 3D shape retrieval (IBSR), where we seek to find the most matched shape of a given single 2D image from a shape repository. Most of the existing works  learn to embed 2D images and 3D shapes into a common feature space and perform metric learning using a triplet loss. Inspired by the great success in recent contrastive learning works on self-supervised  representation learning, we propose a novel IBSR pipeline leveraging contrastive learning. We note that adopting such cross-modal contrastive learning between 2D images and 3D shapes into IBSR tasks is non-trivial and challenging: contrastive learning requires very strong data augmentation in constructed positive pairs to learn the feature invariance, whereas traditional metric learning works do not have this requirement. However, object shape and appearance are entangled in 2D query images, thus making the learning task more difficult than contrasting single-modal data. To mitigate the challenges, we propose to use multi-view grayscale rendered images from the 3D shapes as a shape representation. We then introduce a strong data augmentation technique based on color transfer, which can significantly but naturally change the appearance of the query image, effectively satisfying the need for contrastive learning. Finally, we propose to incorporate a novel category-level contrastive loss that helps distinguish similar objects from different categories, in addition to classic instance-level contrastive loss. Our experiments demonstrate that our approach achieves the best performance on all the three popular IBSR benchmarks, including Pix3D, Stanford Cars, and Comp Cars, outperforming the previous state-of-the-art from 4% - 15% on retrieval accuracy.



## About this repository

This repository provides **data**, **pre-trained models** and **code**.




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

- [Oct 1, 2021] Preliminary version of Data and Code released. For more code and data, coming soon. Please follow our updates.


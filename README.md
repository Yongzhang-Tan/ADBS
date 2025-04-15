# Adaptive Decision Boundary for Few-Shot Class-Incremental Learning（ADBS）
The code repository for "Adaptive Decision Boundary for Few-Shot Class-Incremental Learning" [paper] (AAAI 2025) in PyTorch. If you use the code in this repo for your work, please cite the following bib entries:
```
@inproceedings{li2025adaptive,
  title={Adaptive Decision Boundary for Few-Shot Class-Incremental Learning},
  author={Li, Linhao and Tan, Yongzhang and Yang, Siyuan and Cheng, Hao and Dong, Yongfeng and Yang, Liang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={17},
  pages={18359--18367},
  year={2025}
}
```

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) aims to continuously learn new classes from a limited set of training samples without forgetting knowledge of previously learned classes. Conventional FSCIL methods typically build a robust feature extractor during the base training session with abundant training samples and subsequently freeze this extractor, only fine-tuning the classifier in subsequent incremental phases. However, current strategies primarily focus on preventing catastrophic forgetting, considering only the relationship between novel and base classes, without paying attention to the specific decision spaces of each class. To address this challenge, we propose a plug-and-play Adaptive Decision Boundary Strategy (ADBS), which is compatible with most FSCIL methods. Specifically, we assign a specific decision boundary to each class and adaptively adjust these boundaries during training to optimally refine the decision spaces for the classes in each session. Furthermore, to amplify the distinctiveness between classes, we employ a novel inter-class constraint loss that optimizes the decision boundaries and prototypes for each class. Extensive experiments on three benchmarks, namely CIFAR100, miniImageNet, and CUB200, demonstrate that incorporating our ADBS method with existing FSCIL techniques significantly improves performance, achieving overall state-of-the-art results. 

<img src='imgs/teaser.png' width='500' height='348'>

## Pipline
The whole learning pipline of our model:

<img src='imgs/pipeline.png' width='900' height='453'>

## Results
<img src='imgs/results.png' width='900' height='280'>

Please refer to our [paper] for detailed values.

## Requirements
- [PyTorch >= version 1.1 and torchvision](https://pytorch.org)
- tqdm

## Datasets
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.

## Code Structures
There are five parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: Images and splits for the data sets.
- `dataloader`: Dataloader of different datasets.
- `checkpoint`: The weights and logs of the experiment.

## Training scripts

- CIFAR100

  ```
  python train.py -dataroot YOURDATAROOT -project adbs -dataset cifar100 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.01 -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0 -margin
  ```
  
- CUB200
    ```
  python train.py -dataroot YOURDATAROOT -project adbs -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.002 -lr_new 0.01 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 60 80 100  -gpu '0' -epochs_new_train 1 -margin
    ```

- miniImageNet
    ```
  python train.py -dataroot YOURDATAROOT -project adbs -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 40 70 100  -gpu '0' -epochs_new_train 6 -margin  
    ```
Remember to change `YOURDATAROOT` into your own data root.

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [SAVC](https://github.com/zysong0113/SAVC)
- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [FSCIL](https://github.com/xyutao/fscil)
- [ALICE](https://github.com/CanPeng123/FSCIL_ALICE)
- [OrCo](https://github.com/noorahmedds/OrCo)

## Contact
If there are any questions, please feel free to contact with the author: Yongzhang Tan (tanyongzhang@foxmail.com). 











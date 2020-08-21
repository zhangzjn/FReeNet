## FReeNet &mdash; Official PyTorch Implementation

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic) ![License MIT](https://img.shields.io/github/license/zhangzjn/APB2Face)

Official pytorch implementation of the paper "[FReeNet: Multi-Identity Face Reenactment, CVPR'20](https://arxiv.org/pdf/1905.11805.pdf)".

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.5.1` and `CUDA 10.1` on `Ubuntu 16.04`. 


```shell
# Install python3 packages
pip3 install -r requirements.txt
```

### Datasets in the paper
- Download [RaFD dataset](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=faq) to `datasets/RaFD`.
- Use [Face++ API](https://www.faceplusplus.com.cn/sdk/face-landmarks/) to extract the landmark with 106 points for each face, and save corresponding information to `landmakr.txt`, e.g. `datasets/RaFD/RaFD90/landmark.txt`.

1. Preprocess RaFD dataset.
   ```shell
   > Split RaFD dataset to different dirs based on angle, e.g. RaFD45/image, RaFD90/image, RaFD135/image.
   python3 1-preprocess.py
   ```

### Unified Landmark Converter
1. Train `ULC` model.

   ```shell
   cd src/ULC
   python3 train.py --data RaFD --name RaFD --save_every 10 --every 60 --epochs 200  # for RaFD dataset
   ```

2. Test `ULC` model.

   ```shell
   cd src/ULC
   python3 test.py --data RaFD --name RaFD --save_every 10 --every 60 --epochs 200 --resume  # for RaFD dataset
   ```

### Geometry-aware Generator
1. Train `GAG` model.

   ```shell
   cd src
   python3 train.py --name RaFD-08-21 --gpu_ids 0 --model landmark_L64_Tri --netG resnet_9blocks_cat --dataset_mode RaFD90L64Tri --batch_size 3  # for RaFD dataset
   ```

2. Test `GAG` model.

   ```shell
   cd src
   python3 test_image.py --name RaFD-08-21 --gpu_ids 0 --model landmark_L64_Tri --netG resnet_9blocks_cat --dataset_mode RaFD90L64Tri  # for RaFD dataset
   ```


### Citation
If our work is useful for your research, please consider citing:

```
@inproceedings{zhang2020freenet,
  title={FReeNet: Multi-Identity Face Reenactment},
  author={Zhang, Jiangning and Zeng, Xianfang and Wang, Mengmeng and Pan, Yusu and Liu, Liang and Liu, Yong and Ding, Yu and Fan, Changjie},
  booktitle={CVPR},
  pages={5326--5335},
  year={2020}
}
```

### Acknowledgements

We thank for the source code from the great work [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

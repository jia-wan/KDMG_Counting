# Kernel-based Density Map Generation for Dense Object Counting

## Data preparation
The dataset can be constructed followed by [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).

## Pretrained model
The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TaY5I1eHIt7pm2YBfqw4BfnpX2l3Bof4?usp=sharing).

## Test
``
python test.py --net vgg19 --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
python test.py --net csrnet --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT --resize True
```

## Train
``
python train.py --net vgg19 --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

### Citation
If you use our code or models in your research, please cite with:
``
@inproceedings{wan2019adaptive,
  title={Adaptive density map generation for crowd counting},
  author={Wan, Jia and Chan, Antoni},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1130--1139},
  year={2019}
}

@article{wan2020kernel,
  title={Kernel-based Density Map Generation for Dense Object Counting},
  author={Wan, Jia and Wang, Qingzhong and Chan, Antoni B},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```

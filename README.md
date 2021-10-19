# Shape Generation and Completion Through Point-Voxel Diffusion

[Project]() | [Paper]() 

Implementation of 

## Pretrained Models

Pretrained models can be accessed [here](https://www.dropbox.com/s/a3xydf594fzaokl/cifar10_pretrained.rar?dl=0).

## Requirements:

Make sure the following environments are installed.

```
python==3.6
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
```
The code was tested on Unbuntu with Titan RTX. 


## Training on CIFAR-10:

```bash
$ python train_cifar.py
```

Please refer to the python file for optimal training parameters.

## Results

Some generative results are as follows.
<p float="left">
  <img src="example/cifar_gen.png" width="300"/>
  <img src="example/lsun_gen.png" width="300"/>
</p>



## Reference

```
@inproceedings{han2020joint,
  title={Joint Training of Variational Auto-Encoder and Latent Energy-Based Model},
  author={Han, Tian and Nijkamp, Erik and Zhou, Linqi and Pang, Bo and Zhu, Song-Chun and Wu, Ying Nian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7978--7987},
  year={2020}
}
```

## Acknowledgement

For any questions related to codes and experiment setting, please contact Linqi (Alex) Zhou (alexzhou907@gmail.com). For questions related to model and algorithm in the paper, please contact Tian Han (hantian@ucla.edu). Thanks to [@Tian Han ](https://github.com/hthth0801?tab=repositories) and [@Erik Njikamp](https://github.com/enijkamp) for their colloboration and guidance.
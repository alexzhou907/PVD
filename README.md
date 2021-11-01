# Shape Generation and Completion Through Point-Voxel Diffusion
<p float="left">
  <img src="assets/pvd_teaser.gif" width="80%"/>
</p>

[Project](https://alexzhou907.github.io/pvd) | [Paper](https://arxiv.org/abs/2104.03670) 

Implementation of Shape Generation and Completion Through Point-Voxel Diffusion

[Linqi Zhou](https://alexzhou907.github.io), [Yilun Du](https://yilundu.github.io/), [Jiajun Wu](https://jiajunwu.com/)

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
trimesh=3.7.12
scipy==1.5.1
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

The code was tested on Unbuntu with Titan RTX. 

## Data

For generation, we use ShapeNet point cloud, which can be downloaded [here](https://github.com/stevenygd/PointFlow).

For completion, we use ShapeNet rendering provided by [GenRe](https://github.com/xiumingzhang/GenRe-ShapeHD).
We provide script `convert_cam_params.py` to process the provided data.

For training the model on shape completion, we need camera parameters for each view
which are not directly available. To obtain these, simply run 
```bash
$ python convert_cam_params.py --dataroot DATA_DIR --mitsuba_xml_root XML_DIR
```
which will create `..._cam_params.npz` in each provided data folder for each view.

## Pretrained models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1Q7aSaTr6lqmo8qx80nIm1j28mOHAHGiM?usp=sharing).

## Training:

```bash
$ python train_generation.py --category car|chair|airplane
```

Please refer to the python file for optimal training parameters.

## Testing:

```bash
$ python train_generation.py --category car|chair|airplane --model MODEL_PATH
```

## Results

Some generative results are as follows.
<p float="left">
  <img src="assets/cifar_gen.png" width="300"/>
  <img src="assets/lsun_gen.png" width="300"/>
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
# Shape Generation and Completion Through Point-Voxel Diffusion
<p align="center">
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

Some generation and completion results are as follows.
<p align="center">
  <img src="assets/gen_comp.gif" width="60%"/>
</p>

Multimodal completion on a ShapeNet chair.
<p align="center">
  <img src="assets/mm_shapenet.gif" width="80%"/>
</p>


Multimodal completion on PartNet.
<p align="center">
  <img src="assets/mm_partnet.gif" width="80%"/>
</p>


Multimodal completion on two Redwood 3DScan chairs.
<p align="center">
  <img src="assets/mm_redwood.gif" width="80%"/>
</p>

## Reference

```
@inproceedings{Zhou_2021_ICCV,
    author    = {Zhou, Linqi and Du, Yilun and Wu, Jiajun},
    title     = {3D Shape Generation and Completion Through Point-Voxel Diffusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5826-5835}
}
```

## Acknowledgement

For any questions related to codes and experiment setting, please contact [Linqi Zhou](linqizhou@stanford.edu) and [Yilun Du](yilundu@mit.edu). 

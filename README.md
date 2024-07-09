# Neural Refinement for Absolute Pose Regression with Feature Synthesis
**[Shuai Chen](https://scholar.google.com/citations?user=c0xTh_YAAAAJ&hl=en), 
[Yash Bhalgat](https://scholar.google.com/citations?user=q0VSEHYAAAAJ&hl=en),
[Xinghui Li](https://scholar.google.com/citations?user=XLlgbBoAAAAJ&hl=en), 
[Jiawang Bian](https://scholar.google.com/citations?user=zeGz5JcAAAAJ&hl=en&oi=sra),
[Kejie Li](https://scholar.google.com/citations?hl=en&user=JBwsoCUAAAAJ),
[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), 
and [Victor Prisacariu](https://scholar.google.com/citations?user=GmWA-LoAAAAJ&hl=en) (CVPR 2024)**

**[Project Page](https://nefes.active.vision) | [Paper](https://arxiv.org/abs/2303.10087)**

[![NeFeS1](imgs/pipeline.png)](https://arxiv.org/abs/2303.10087)
[![NeFeS2](imgs/nefes.png)](https://arxiv.org/abs/2303.10087)


## Installation
We tested our code based on CUDA11.3+, PyTorch 1.11.0+, and Python 3.7+ using [docker](https://docs.docker.com/engine/install/ubuntu/).

We also provide a `conda` environment
```sh
conda env create -f environment.yml
conda activate nefes
pip install git+https://github.com/princeton-vl/lietorch.git # if your lietorch doesn't work, you can set lietorch=False in poses.py
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# install pytorch3d
cd ..
git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && pip install -e .
```

## Datasets
This paper uses two public datasets:
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)

- **7-Scenes**

We use a similar data preparation as in [MapNet](https://github.com/NVlabs/geomapnet). You can download the [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) datasets to the `data/deepslam_data/7Scenes` directory using the script below.

```sh
cd data
python setup_7scenes.py
```

1. we additionally computed a pose averaging stats (pose_avg_stats.txt) and manually tuned world_setup.json in `data/7Scenes` to align the 7Scenes' coordinate system with NeRF's coordinate system (OpenGL). You could generate your own re-alignment to a new pose_avg_stats.txt using the `--save_pose_avg_stats` configuration.

2. In our `setup_7scenes.py` script, we also copy the 7scenes colmap poses to the deepslam_data/7Scenes/{SCENE}/ folder, courtsey to [Brachmann21](https://github.com/tsattler/visloc_pseudo_gt_limitations).

- **Cambridge Landmarks**

To downlaod Cambridge Landmarks, please use this script.
```sh
cd data
python setup_cambridge.py
```
We also put the `pose_avg_stats.txt` and `world_setup.json` to the `data/Cambridge/CAMBRIDGE_SCENES` like we provided in the source code. 

As we described in the paper, we also applied semantic filtering when training NeFeS to filter out temporal objects using [Cheng22](https://github.com/facebookresearch/Mask2Former). Therefore, in the script, we download and put them into `data/Cambridge/{CAMBRIDGE_SCENE}/train/semantic` and `data/Cambridge/{CAMBRIDGE_SCENE}/test/semantic`.

## Pre-trained Models
We currently provide pretrained NeFeS models and DFNet models used in our paper. 

Download and decompress [paper_models.zip](https://www.robots.ox.ac.uk/~shuaic/NeFeS2024/paper_models.zip) to {REPO_PATH}/logs/paper_models
```sh
wget https://www.robots.ox.ac.uk/~shuaic/NeFeS2024/paper_models.zip
unzip paper_models.zip
mkdir logs
mv paper_models/ logs/
```

### GPUs for Pre-trained Models and Verifying Paper Results
Due to our limited resource, my pre-trained models are trained using different GPUs such as Nvidia 3090, 3080ti, RTX 6000, or 1080ti GPUs. We noticed that models' performance might jitter slightly (could be better or worse) when running inference with different types of GPUs. Therefore, all experiments on the paper are reported based on the same GPUs as they were trained. To providing necesssary reference, we also include the experimental results ran by our machines.

#### You can easily obtain our paper results (Table 1 and Table 2 DFNet + NeFeS50) by running:
```sh
sh eval.sh
```

## Training NeFeS
We provide NeFeS training script in `train_nefes.sh`
```sh
sh train_nefes.sh
```
In this script, we run a three stage progressive training schedule, as described in the Supplementary Material of the paper.
```sh
# Stage 1 of training color only nerf, initializing the 3D geometry to a reasonable extent.
python run_nefes.py --config config/7Scenes/dfnet/config_stairs_stage1.txt

# Stage 2 and 3 for training feature and fusion modules, obtaining best neural feature fields performance for NeFeS.
python run_nefes.py --config config/7Scenes/dfnet/config_stairs_stage2.txt
```

## Evaluation
After training NeFeS, it is ready to test the APRs with NeFeS refinement. Notice that we've already provided paper results [above](#you-can-easily-verify-our-paper-results-table-1-and-table-2-dfnet--nefes50-by-running) ran by ourselves.
To use your own trained model, you can choose to use the following script. 
```sh
# this script is an example of running DFNet + NeFeS50
sh test_apr_refinement.sh
```

In the script, we utilize paper models by default in the config file. You could replace the default models with your own models if you have trained ones.
```sh
python test_refinement.py --config config/7Scenes/dfnet/config_stairs_DFM.txt --ft_path $YOUR_NeFeS
```

If your GPU is out-of memory, please consider reducing `--netchunk` parameters.

If you want to try to see if NeFeS can refine your own APR model/pose estimator, you can add your network loader to `load_APR_and_FeatureNet()` in dm/direct_pose_model.py.
Notice that it is recommanded to train your APR/pose estimator in openGL coordinate system (best way is through our dataloader, as we did for [PoseNet (pytorch)](https://github.com/ActiveVisionLab/direct-posenet/tree/main) and [MsTransformer](https://github.com/yolish/multi-scene-pose-transformer)). This is because our NeFeS is trained in openGL convention, otherwise you will have to adjust the cooridnate system yourself.

## Acknowledgement
We thank Dr. Michael Hobley and Dr. Theo Costain for their generous discussion on this work as well as their kind proof reading for our paper manuscripts. We also thank Changkun Liu for kindly providing assistant on ensuring conda environment consistency.

## Publications
Please cite our paper and star this repo if you find our work helpful. Thanks!
```
@inproceedings{chen2024nefes,
  author    = {Chen, Shuai and Bhalgat, Yash and Li, Xinghui and Bian, Jia-Wang and Li, Kejie and Wang, Zirui and Prisacariu, Victor Adrian},
  title     = {Neural Refinement for Absolute Pose Regression with Feature Synthesis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {20987-20996}
}
```
This code builds on previous camera relocalization pipelines, namely Direct-PoseNet and DFNet. Please consider citing:
```
@inproceedings{chen2022dfnet,
  title={DFNet: Enhance Absolute Pose Regression with Direct Feature Matching},
  author={Chen, Shuai and Li, Xinghui and Wang, Zirui and Prisacariu, Victor},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
@inproceedings{chen2021direct,
  title={Direct-PoseNet: Absolute pose regression with photometric consistency},
  author={Chen, Shuai and Wang, Zirui and Prisacariu, Victor},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={1175--1185},
  year={2021},
  organization={IEEE}
}
```

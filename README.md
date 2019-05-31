
# Carla Data Export
This repository contains the code we used for generating training data from the CARLA simulator in our paper [**Multimodal 3D Object Detection from Simulated Pretraining**](https://arxiv.org/abs/1905.07754)

## Installation
Download and extract CARLA 0.8.4 from https://github.com/carla-simulator/carla/releases/tag/0.8.4  
This project expects the carla folder to be inside this project i.e PythonClient/carla-data-export/carla

## Generate data
Start a CARLA server (the export tool is by default tuned for 10fps simulation)  
Start datageneration by running gen_data.sh

## Citation
If you have used our work, please cite our paper:
```
@article{brekke2019multimodal,
  title={Multimodal 3D Object Detection from Simulated Pretraining},
  author={Brekke, {\AA}smund and Vatsendvik, Fredrik and Lindseth, Frank},
  journal={arXiv preprint arXiv:1905.07754},
  year={2019}
}
```

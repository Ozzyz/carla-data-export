
# Carla Data Export
This repository contains the code we used for generating training data from the CARLA simulator in our paper [**Multimodal 3D Object Detection from Simulated Pretraining**](https://arxiv.org/abs/1905.07754)

## Installation
Download and extract CARLA 0.8.4 from https://github.com/carla-simulator/carla/releases/tag/0.8.4  
This project expects the carla folder to be inside this project i.e PythonClient/carla-data-export/carla  
Install all the necessary requirements for your python environment using:
```
pip install -r requirements.txt
```

# Generating data
Before the data generation scripts can be run you must start a CARLA server. This can be done by running the executable in the CARLA root folder with the appropriate parameters. Running the server on windows in a small 200x200 window would for example be:
```
./CarlaUE4.exe -carla-server -fps=10 -windowed -ResX=200 -ResY=200
```
Once the server is running, data generation can be started using (remove --autopilot for manual control):
```
python datageneration.py --autopilot
```

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

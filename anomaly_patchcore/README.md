# Anomaly Detection
- This repo contains the source code for doing feature extraction and anomaly detection. Training is done using the anomalib. Instructions will be provided below



## Installation
- git clone this repo to your ROS workspace (or into both Xavier & SBC) and checkout to POC-anomaly-detection-final
- ensure you have bx_msgs repo
- catkin build
- create a models folder in the same directory as /src to store all your models
- For your local PC: before running, ensure you have a feature_extractor model and a memory_bank (from training) present. In my case, my feature_extractor is a Wideresnet_Timms.pt wrapped with Timm Feature Extractor. Change the shebang of feature_extractor.py to be your venv python as it is running on a lower pytorch version (like the Xavier's) from your local. If your local is running the same version as Xavier, you may just change it to #!/usr/bin/python3

- For Xavier: before running, ensure you have a feature_extractor model. Ensure shebang is **#!/usr/bin/python3**
- For SBC: before running, ensure you have a memory_bank (from training) present. Then **pip3 install opencv-python == 4.5**
- To run:
```
roslaunch anomaly_inference feature_extractor.launch model:=Wideresnet_Timms.pt

roslaunch anomaly_inference inference_KNN.launch model:=model_610_memory_bank.npy
```


## Training
- Create a python 3.10 or above venv. Activate the env. Make sure you're in your venv when running any anomalib related scripts.

- Install the anomalib library in a <separate_directory> (eg. Documents). You can follow the instructions in the "Development Install" section of https://github.com/openvinotoolkit/anomalib

- copy the training script in <separate_directory>/anomaly_inference/scripts/anomalib_scripts/training_anomaly_recog.py to the same directory as where you installed anomalib library. Change the directory of the paths in the script to your dataset directory.

- To train more images, replace the script in the anomalib library: anomalib/src/anomalib/callbacks/metrics.py with this script ../anomaly_inference/scripts/anomalib_scripts/torch_model.py. PatchCore model modified to use annoy library during training process for more efficient Nearest-Neighbor search.

- replace the script in the anomalib library: /home/beex/Documents/ML Task/anomalib/src/anomalib/models/image/patchcore/torch_model.py with this script ../anomaly_inference/scripts/anomalib_scripts/metrics.py. You can get a tensor values of Precision & Recall scores which you can collect and use to calculate a proper F1 score manually.

- python3 <separate_directory>/anomaly_inference/scripts/anomalib_scripts/training_anomaly_recog.py

- To convert your .ckpt model to .pt model, run:
anomalib export --model Patchcore --export_type torch --ckpt_path results/Patchcore/confirmed_resnet_change2/latest/weights/lightning/model.ckpt

- To save memory bank:
```
  ## change the path in inference_memory_bank_saver.py to load your .pt model
  - rosrun anomaly_inference inference_memory_bank_saver.py
```
- To save feature extractor:

  - uncomment these lines in feature_extractor.py:
```
# from utils.Feature_Extractor import TimmFeatureExtractor  # Custom module

# # Use PatchCore's Feature Extractor (commented out)
# self.feature_extractor = (
#     TimmFeatureExtractor(backbone=self.backbone, layers=self.layers, pre_trained=True).eval().to(self.device)
# )
# scripted_model = torch.jit.script(self.full_model)
# torch.jit.save(scripted_model)
```
```
  - rosrun anomaly_inference feature_extractor.launch
```
## Running static Inference with anomalib, ALL paths are wrt to where you are currently at in the terminal
python3 "/home/beex/Documents/ML Task/anomalib/tools/inference/lightning_inference.py" --model anomalib.models.Cfa --ckpt_path results/Cfa/official_run1/v3/weights/lightning/model.ckpt --data.path interview_testset/masks --output ./outputs




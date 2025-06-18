# Anomaly Detection

## IMPORTANT
**OPEN THIS README IN VSCODE and press CTRL+SHIFT+V**

## Installation
If you're seeing this, you should have downloaded the entire ML_anomaly_pipeline folder from onedrive
   ```bash
  conda create -n anamoly_detection python=3.10
  conda activate anamoly_detection
  cd ML_anomaly_pipeline/anomalib  (or wherever you copied ML_anomaly_pipeline to)
  pip install -e .
  cd ..
  run **./dependencies.sh**
  ```
## Data prep
- Cartesian sonar images would be used for detection to maintain linearity throughout frames

   ```bash
   cd image_extraction_rosbags
   ```
   switch to an environment that has ROS
   ```bash
   python3 "main.py" --use_sonar_as_primary True --input_folder "<rosbags_directory>" --output_folder "<directory_to_save_images>" --threshold 0.99
   ```
  - make sure to adjust the parameters accordingly
  - script will extract sonar and camera images and cos filter out duplicates
  - **IMPORTANT:** ensure that the threshold set should only remove at most 1/3 of the pictures. More than that may result in FPs in many frames, in complex environments (eg. blyth and jacket)
  - ideally run this script on a rosbag with NO anomalous frames & another time on a rosbag with anomalous sonar frames to get your **normal** & **abnormal** data separately
  - extract out all the normal/abnormal sonar images accordingly into separate folders
  - lastly, run **img_crop.py** to crop **BOTH** normal and abnormal Cartesian Images to 398 by 398. Make sure to change the paths accordingly in the script below
  ```bash
  python3 img_crop.py
  ```


## Training

- **IMPORTANT:** to allow more training of images, training has been shifted over to CPU. Therefore, take note when opening CPU-intensive applications (eg. multiple VSCode windows, Chatgpt, background ROS programs etc), it could cause the training to crash
- To **allow more images training & ensure consistent Neighbor Search** with our inference, the PatchCore script in the anomalib library has been replaced to use annoy library for nearest neighbour search for now. Coreset has also been stored in cpu instead of VRAM to allow for more images to be trained. This ensures consistency with our inference too:
  - PatchCore model (<where_you_clone_anomalib_repo>/anomalib/src/anomalib/models/image/patchcore/torch_model.py) modified to use annoy library during training process for more efficient Nearest-Neighbor search.
  - changes have been made to <where_you_clone_anomalib_repo>/anomalib/src/anomalib/models/image/patchcore/lightning_model.py and torch_model.py to train on more images


Run training script, **training_anomaly_recog.py**,
```bash
python3 <where_you_clone_anomalib_folder>/training_anomaly_recog.py
```
- It is under same directory as anomalib folder:
- Change the directory of the dataset paths in the script to your dataset directory. Follow the comments inside on how to set your dataset directories.
- **results** will be saved in the directory where you ran the training script


Run the below command at the same directory where you run the training to **convert your .ckpt model to .pt model**,
```bash
anomalib export --model Patchcore --export_type torch --ckpt_path results/Patchcore/confirmed_resnet_change2/latest/weights/lightning/model.ckpt
```
- .pt model will be saved under /weights in your /results directory


To save memory bank:
```bash
  rosrun anomaly_inference inference_memory_bank_saver.py

  # Make sure to give execute permissions to files inside the ros package

  # Ensure to change the shebang of your inference_memory_bank_saver.py script to point it to the conda or system python version

```
  - Save the .pt model in the /models dir in your anomaly_inference
  - change the model in inference_memory_bank_saver.py from your ros package under scripts folder to load your .pt model
  - memory bank would be saved in your models dir of your rospackage

To find correct threshold for inference and f1 score:
  - copy the Precision, Recall and Threshold values printed on the terminal and paste them accordingly to the script, **f1_score_calculator.py**
```bash
python3 f1_score_calculator.py
```

## Save Wideresnet feature extractor
**SAVING FEATURE EXTRACTOR IS NOT NECESSARY FOR NOW**

To save feature extractor:
  - uncomment these lines in feature_extractor.py:
```bash
# from utils.Feature_Extractor import TimmFeatureExtractor  # Custom module

# # Use PatchCore's Feature Extractor (commented out)
# self.feature_extractor = (
#     TimmFeatureExtractor(backbone=self.backbone, layers=self.layers, pre_trained=True).eval().to(self.device)
# )
# scripted_model = torch.jit.script(self.full_model)
# torch.jit.save(scripted_model)
```

## To start inference with your model
```bash
roslaunch anomaly_inference feature_extractor.launch model:=<absolute path to your model>

roslaunch anomaly_inference inference_KNN.launch model:=<absolute path to your model>
```
- Run your bag file
- Use rqt to view


## Further development
- there are commented out code blocks in **anomalib/src/anomalib/models/image/patchcore/torch_model.py** and **anomalib/src/anomalib/models/image/patchcore/lightning _model.py** that integrated yolo11 segmentation jit model in as a feature extractor. You may comment them out and run the training accordingly to see the results



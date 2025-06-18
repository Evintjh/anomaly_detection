# Import the required modules
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Import the datamodule
from anomalib.data import Folder
# from anomalib.data.utils import TestSplitMode

# Create the datamodule
# Give a meaningful name
# root == where your normal & abnormal image directory will reside in
datamodule = Folder(
    name="anomaly_training_Cart_images_blyth_only_OVERFIT",
    # root="/mnt/d/FullData/Normal_seabed_with_no_anomaly/tuas pipeline aug 2024/POLAR",
    # normal_dir="testing",
    # abnormal_dir="abnormal_test",

    # normal_dir="tuas_dataset_resized_1700",
    # abnormal_dir="abnormal",

    ## Polar Blyth Final
    # root = "/mnt/d/FullData/Normal_seabed_with_no_anomaly/blyth",
    # normal_dir = "normal_filtered_final",
    # # normal_dir = "tuas_dataset_resized_1700",
    # abnormal_dir = "abnormal",

    ## Cartesian Blyth
    root = "/mnt/d/FullData/Normal_seabed_with_no_anomaly/blyth/cartesian",
    normal_dir = "final_normal_again_no_aug_PROPER",
    # normal_dir = "tuas_dataset_resized_1700",
    abnormal_dir = "abnormal_cropped",
    # train_batch_size=1,
)

# Setup the datamodule
datamodule.setup()
print(f"Training batch size: {datamodule.train_dataloader().batch_size}")

# Training
model = Patchcore()
engine = Engine(max_epochs=1)


# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

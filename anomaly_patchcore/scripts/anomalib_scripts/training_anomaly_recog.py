# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore, EfficientAd, Cfa, Csflow, Draem, Stfpm
from anomalib.engine import Engine

# Import the datamodule
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
    name="embeddings_check",
    root="/mnt/d/FullData/Normal_seabed_with_no_anomaly/tuas pipeline aug 2024",
    normal_dir="L1_loss_tester",
    abnormal_dir="abnormal_resized",
    task="classification",
    train_batch_size=8,
)

# Setup the datamodule
datamodule.setup()
print(f"Training batch size: {datamodule.train_dataloader().batch_size}")

# Training
model = Patchcore()
engine = Engine(task="classification", max_epochs=30)

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

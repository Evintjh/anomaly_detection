#!/usr/bin/python3

### Use a virtual environment (venv) if your local PyTorch version != 1.10.0
### Xavier torch == 1.10.0, so ensure your venv torch version is similar.

# ######################### PATCHCORE TIMM JIT model in real-time ROS ####################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rospy
from PIL import Image as PILImage
from io import BytesIO
import rospkg
from sensor_msgs.msg import CompressedImage
from bx_msgs.msg import SurveyorInfo, Anomaly, Utm
from nav_msgs.msg import Odometry
import math
import cv2

# from utils.Feature_Extractor import TimmFeatureExtractor  # Custom module


class FeatureExtractorROS(nn.Module):
    """
    A ROS-integrated feature extractor using a pre-trained TorchScript (JIT) model.
    - Subscribes to polar sonar image data from `/ikan/fls/data` for metadata
    - Subscribes to Cartesian sonar image data from `/ikan/sonar/image/compressed` for processing
    - Processes images using a deep learning model
    - Publishes extracted feature embeddings to `/anomaly/feature_embeddings`
    """

    def __init__(self, backbone="wide_resnet50_2", layers=None, model_path=None):
        super(FeatureExtractorROS, self).__init__()

        # Set computation device. Cuda not compatible for now as torch model was saved with torch==1.8.0, which is not compatible with 3060 nvidia kernel (CPU by default, as this is a real-time ROS node)
        self.device = torch.device("cuda")
        rospy.loginfo(f"Using device: {self.device}")

        # Model configuration
        self.layers = layers if layers else ["layer2", "layer3"]  # Define feature extraction layers
        self.full_model = torch.jit.load(model_path, map_location=self.device).eval()  # Load pre-trained JIT model

        # Feature Pooling layer (Same as PatchCore)
        self.feature_pooler = nn.AvgPool2d(3, 1, 1).to(self.device)

        self.sync_time = rospy.Time  # Stores the timestamp of the latest sonar image
        self.odom_timestamp_ms = rospy.Time
        self.count = 0  # Counter for downsampling
        self.sonar_range = 0  # Store sonar range from polar image data
        # ROS Setup: Subscribe to both polar and Cartesian image topics
        self.utm_sub = rospy.Subscriber("/ikan/nav/utm", Utm, self.utm_callback, queue_size=1)
        self.alt_sub = rospy.Subscriber("/ikan/pathfinder/altitude", Odometry, self.alt_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/ikan/nav/world_ned", Odometry, self.odom_callback, queue_size=1)
        self.fls_sub = rospy.Subscriber("/ikan/fls/data", SurveyorInfo, self.fls_callback, queue_size=1)
        self.cart_sub = rospy.Subscriber("/ikan/sonar/image/compressed", CompressedImage, self.cart_callback, queue_size=1)
        self.embeddings_pub = rospy.Publisher("/anomaly/feature_embeddings", Anomaly, queue_size=10)

        self.utm_x = 0
        self.utm_y = 0
        self.altitude = 0
        self.baselink_easting = 0
        self.baselink_northing = 0
        self.baselink_depth_msl = 0
        self.baselink_depth_surface = 0
        self.baselink_altitude = 0
        self.baselink_yaw = 0
        self.baselink_pitch = 0
        self.baselink_roll = 0

        rospy.loginfo("FeatureExtractorROS Node Initialized")

    def quaternion_to_euler_angle(self, w, x, y, z):
        """
        Convert a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees.
        """
        ysqr = y * y
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1) * 180.0 / math.pi
        t2 = 2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch = math.asin(t2) * 180.0 / math.pi
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4) * 180.0 / math.pi
        if yaw < 0:
            yaw += 360.0
        return roll, pitch, yaw

    def alt_callback(self, msg):
        self.altitude = msg.pose.pose.position.z

    def utm_callback(self, msg):
        self.utm_x = msg.x
        self.utm_y = msg.y

    def odom_callback(self, msg):
        self.odom_timestamp_ms = msg.header.stamp.to_sec() * 1000
        self.baselink_northing = msg.pose.pose.position.x
        self.baselink_easting = msg.pose.pose.position.y
        self.baselink_depth_msl = msg.pose.pose.position.z
        self.baselink_depth_surface = self.baselink_depth_msl
        self.baselink_altitude = self.altitude
        roll, pitch, yaw = self.quaternion_to_euler_angle(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        )
        self.baselink_yaw = yaw
        self.baselink_pitch = pitch
        self.baselink_roll = roll

    def fls_callback(self, msg: SurveyorInfo):
        """
        Callback function triggered when a new polar sonar image is received.
        - Updates metadata (e.g., timestamp, sonar range)
        """
        try:
            self.sync_time = msg.unix_time_ms  # Store timestamp (milliseconds)
            self.sonar_range = msg.max_range  # Store sonar range
            print("Time diff: ", abs(msg.unix_time_ms - self.odom_timestamp_ms))

        except Exception as e:
            rospy.logerr(f"Error processing polar image metadata: {e}")

    def cart_callback(self, msg: CompressedImage):
        """
        Callback function triggered when a new Cartesian sonar image is received.
        - Decodes the compressed image data to a NumPy array
        - Applies downsampling and runs inference pipeline if condition met
        """
        try:
            # Decode the compressed image using PIL
            image_buffer = BytesIO(msg.data)
            image = PILImage.open(image_buffer).convert("RGB")
            image = np.array(image)

            # Downsampling logic: Process every 8th image
            self.count += 1
            if self.count == 5:
                # Run inference on the extracted Cartesian image
                self.infer(image)
                self.count = 0

        except Exception as e:
            rospy.logerr(f"Error processing Cartesian image: {e}")

    def pre_process_cart(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepares the Cartesian image for the model:
        - Crops a 398x398 square centered on the image
        - Resizes it to 256x256
        - Crops a 224x224 square centered on the resized image
        - Converts from HWC to CHW format
        - Normalizes pixel values (0-255 → 0-1)
        """
        # Get image dimensions
        h, w = image.shape[:2]
        crop_size = 398

        # Calculate the center and crop boundaries
        center_x, center_y = w // 2, h // 2
        start_x = max(0, center_x - crop_size // 2)
        start_y = max(0, center_y - crop_size // 2)
        end_x = min(w, start_x + crop_size)
        end_y = min(h, start_y + crop_size)

        # Crop the image
        cropped_image = image[start_y:end_y, start_x:end_x]
        # cropped_image_original = cropped_image

        # Resize to 256x256
        cropped_image = PILImage.fromarray(cropped_image).resize((256, 256))
        cropped_image = np.array(cropped_image)

        # Center crop to 224x224
        cropped_image = PILImage.fromarray(cropped_image)
        cropped_image = cropped_image.crop(((256 - 224) // 2, (256 - 224) // 2, (256 + 224) // 2, (256 + 224) // 2))
        cropped_image = np.array(cropped_image)
        cropped_image_original = cropped_image


        if cropped_image.ndim == 3:
            cropped_image = np.transpose(cropped_image, (2, 0, 1))  # Convert from HWC to CHW format

        cropped_image = cropped_image / 255.0  # Normalize pixel values
        tensor = torch.from_numpy(cropped_image).float().unsqueeze(0).to(self.device)  # Convert to Torch tensor

        return tensor, cropped_image_original

    # def pre_process_cart(self, image: np.ndarray, gamma: float = 1.5):
    #     """
    #     Preprocesses the Cartesian image by:
    #     - Cropping a 398x398 square centered on the image
    #     - Applying bilinear filtering (resizing to same size)
    #     - Applying a power filter (gamma correction)
    #     - Resizing to 224x224
    #     - Normalizing and formatting for the model

    #     Returns:
    #         tensor: Torch tensor [1, C, H, W]
    #         filtered_image: 398x398 filtered image as uint8 RGB (for logging/storage)
    #     """
    #     # Crop to 398x398 center
    #     h, w = image.shape[:2]
    #     crop_size = 398
    #     center_x, center_y = w // 2, h // 2
    #     start_x = max(0, center_x - crop_size // 2)
    #     start_y = max(0, center_y - crop_size // 2)
    #     end_x = min(w, start_x + crop_size)
    #     end_y = min(h, start_y + crop_size)
    #     cropped = image[start_y:end_y, start_x:end_x]

    #     # --- Bilinear Filtering ---
    #     cropped_bilinear = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    #     # --- Power Filter (Gamma Correction) ---
    #     normalized = cropped_bilinear / 255.0
    #     power_filtered = np.power(normalized, gamma)
    #     filtered = (power_filtered * 255).clip(0, 255).astype(np.uint8)

    #     # --- Resize to 224x224 for the model ---
    #     resized = PILImage.fromarray(filtered).resize((224, 224))
    #     resized_np = np.array(resized)

    #     # --- Normalize + Format for PyTorch ---
    #     if resized_np.ndim == 3:
    #         resized_np = np.transpose(resized_np, (2, 0, 1))  # HWC → CHW

    #     resized_np = resized_np / 255.0  # Normalize to [0, 1]
    #     tensor = torch.from_numpy(resized_np).float().unsqueeze(0).to(self.device)  # Add batch dim

    #     return tensor, filtered

    def infer(self, image: np.ndarray):
        """
        Runs the inference pipeline on the received Cartesian image.
        - Preprocesses the image
        - Extracts features using the loaded model
        - Publishes the extracted feature embeddings
        """
        input_tensor, cropped_image_original = self.pre_process_cart(image)
        current_img = cropped_image_original
        # current_img = image  # Use the original Cartesian image for publishing

        # Extract feature embeddings without gradient tracking (for efficiency)
        with torch.no_grad():
            embeddings = self.forward(input_tensor.to(self.device))

        # Publish embeddings to the ROS topic
        self.publish_embeddings(current_img, embeddings)

    def forward(self, input_tensor: torch.Tensor):
        """
        Passes the input image through the model to extract feature embeddings.
        - Uses multiple layers defined in `self.layers`
        - Applies average pooling to the extracted features (PatchCore technique)
        - Combines the features into a final embedding
        """
        with torch.no_grad():
            features = self.full_model(input_tensor.to(self.device))  # Extract features
            features = dict(zip(self.layers, features))  # Store as dictionary

        # Apply feature pooling to each extracted layer
        pooled_features = {layer: self.feature_pooler(features[layer].to(self.device)) for layer in features}

        # Debugging: Print feature shapes after pooling
        for layer, feature in pooled_features.items():
            rospy.loginfo(f"Layer: {layer}, Pooled Shape: {feature.shape}")

        # Debugging: Print feature statistics (min/max values)
        for layer, feature in pooled_features.items():
            rospy.loginfo(f"Layer: {layer}, Min: {feature.min().item()}, Max: {feature.max().item()}")

        # Generate final feature embedding
        embedding = self.generate_embedding(pooled_features)
        rospy.loginfo(f"After embedding generation: Min: {embedding.min().item()}, Max: {embedding.max().item()}")

        return embedding

    def generate_embedding(self, features: dict) -> torch.Tensor:
        """
        Combines feature maps from multiple layers into a single embedding tensor.
        - Uses bilinear interpolation to ensure all feature maps are the same size
        - Concatenates them along the channel dimension
        """
        embeddings = features[self.layers[0]]  # Start with the first feature layer

        for layer in self.layers[1:]:
            layer_embedding = features[layer].to(self.device)

            # Resize feature maps to match the first layer using bilinear interpolation
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")

            # Concatenate feature embeddings along the channel dimension
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    def publish_embeddings(self, image: np.ndarray, embeddings: torch.Tensor):
        """
        Converts extracted embeddings into a ROS `Anomaly` message and publishes it.
        - Flattens the embedding tensor
        - Attaches a timestamp and other metadata from polar image data
        - Publishes the message to `/anomaly/feature_embeddings`
        """
        embeddings_np = embeddings.cpu().detach().numpy().flatten()  # Convert to NumPy and flatten
        data = embeddings_np.tolist()

        # Encode image into JPEG format
        pil_img = PILImage.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        msg = Anomaly()
        msg.name = "anomaly"
        msg.source = 2
        msg.tracker_id = 0
        msg.sensor_setting_index = int(round(self.sonar_range))  # Use sonar range from polar data
        msg.unix_time_ms = self.sync_time  # Use timestamp from polar data
        msg.data.data = data  # Attach feature data
        msg.data_image = list(image_bytes)  # ROS expects list of uint8
        msg.baselink_easting = int(self.baselink_easting * 1000)
        msg.baselink_northing = int(self.baselink_northing * 1000)
        msg.baselink_depth_msl = int(self.baselink_depth_msl * 100)
        msg.baselink_depth_surface = int(self.baselink_depth_surface * 100)
        msg.baselink_altitude = int(self.baselink_altitude * 100)
        msg.baselink_yaw = int(self.baselink_yaw * 100)
        msg.baselink_pitch = int(self.baselink_pitch * 100)
        msg.baselink_roll = int(self.baselink_roll * 100)

        # Publish message
        self.embeddings_pub.publish(msg)


# Main execution
if __name__ == "__main__":
    try:
        # Initialize ROS node
        rospy.init_node("feature_extractor_node")

        # Get the model path from the launch file
        model_path = rospy.get_param("~model")

        print(f"Loading model from: {model_path}")

        # Initialize the feature extractor node
        model = FeatureExtractorROS(model_path=model_path)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down FeatureExtractorROS Node")

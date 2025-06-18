#!/usr/bin/python3

import numpy as np
import rospy
import cv2
from annoy import AnnoyIndex
from bx_msgs.msg import Anomaly


class AnomalyMapGenerator:
    """
    Generates an anomaly map from the patch scores.
    - Uses Gaussian blur to smoothen the map.
    - Resizes patch-based anomaly scores to match the input image resolution.
    """

    def __init__(self, sigma: int = 4) -> None:
        self.sigma = sigma  # Standard deviation for Gaussian blur

    def compute_kernel_size(self, sigma: int):
        """Compute the Gaussian kernel size from the sigma value."""
        truncate = 4
        radius = int(truncate * sigma + 0.5)
        ksize = 2 * radius + 1
        return ksize  # Kernel size must be odd

    def compute_anomaly_map(self, patch_scores: np.ndarray, image_size: tuple = None):
        """
        Generates the anomaly map.
        - If `image_size` is provided, interpolates the patch scores.
        - Applies Gaussian blur for smoothing.
        """
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = self.interpolate(patch_scores, image_size)

        ksize = self.compute_kernel_size(self.sigma)

        return cv2.GaussianBlur(anomaly_map, (ksize, ksize), sigmaX=self.sigma, borderType=cv2.BORDER_REFLECT)

    @staticmethod
    def interpolate(array: np.ndarray, size: tuple) -> np.ndarray:
        """
        Resizes a spatial anomaly score map to match the desired `size`.
        - Uses OpenCV's bilinear interpolation.
        """
        resized_array = cv2.resize(array[0, 0], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

        return resized_array


class AnomalyDetector:
    """
    Anomaly Detector using memory bank and nearest neighbor search.
    - Loads a precomputed memory bank (numpy file).
    - Uses Annoy (Approximate Nearest Neighbors) for fast anomaly scoring.
    - Computes anomaly maps and publishes them via ROS.
    """

    def __init__(self, memory_bank_path, num_neighbors=9):
        self.device = "cpu"  # Running on CPU
        self.memory_bank = np.load(memory_bank_path, allow_pickle=True)  # Load stored feature embeddings
        self.num_neighbors = num_neighbors  # Number of nearest neighbors to consider
        self.anomaly_map_generator = AnomalyMapGenerator()  # Initialize anomaly map generator
        self.baselink_easting = 0
        self.baselink_northing = 0
        self.baselink_depth_msl = 0
        self.baselink_depth_surface = 0
        self.baselink_altitude = 0
        self.baselink_yaw = 0
        self.baselink_pitch = 0
        self.baselink_roll = 0
        print("model loaded")

        # Convert memory bank to NumPy array
        self.memory_bank_np = self.memory_bank
        embedding_size = self.memory_bank_np.shape[1]

        # Initialize Annoy index for fast approximate nearest neighbors
        self.annoy_index = AnnoyIndex(embedding_size, metric="euclidean")

        # Add each stored feature vector into the Annoy index
        for i in range(self.memory_bank_np.shape[0]):
            self.annoy_index.add_item(i, self.memory_bank_np[i])

        self.annoy_index.build(10)  # Build the Annoy tree with 10 trees for fast lookup

        # ROS Subscribers (Listening to feature embeddings)
        self.embeddings_sub = rospy.Subscriber(
            "/anomaly/feature_embeddings", Anomaly, self.embeddings_callback, queue_size=1
        )

        # ROS Publisher (Publishing anomaly maps)
        self.anomaly_map_pub = rospy.Publisher("/anomaly/detection_map", Anomaly, queue_size=10)

        # Store latest sonar timestamp
        self.latest_sonar_time = None

        # Expected shape of embeddings from feature extractor
        self.expected_shape = (1, 1536, 28, 28)

    def embeddings_callback(self, msg: Anomaly):
        """
        Callback for processing incoming feature embeddings.
        - Retrieves the timestamp from the header.
        - Converts the received embeddings into NumPy array.
        - Runs anomaly detection and publishes the anomaly map.
        """
        embedding_time = msg.unix_time_ms  # Timestamp in milliseconds

        # Convert the ROS Float32MultiArrayStamped message into a NumPy array
        embeddings_np = np.array(msg.data.data)
        embedding = embeddings_np.reshape(self.expected_shape)  # Reshape to match expected tensor format

        # Perform anomaly detection
        anomaly_output = self.detect_anomaly(embedding)
        current_sonar_img = msg.data_image

        # Publish the anomaly map with the correct timestamp
        self.sonar_range = msg.sensor_setting_index
        self.baselink_easting = msg.baselink_easting
        self.baselink_northing = msg.baselink_northing
        self.baselink_depth_msl = msg.baselink_depth_msl
        self.baselink_depth_surface = msg.baselink_depth_surface
        self.baselink_altitude = msg.baselink_altitude
        self.baselink_yaw = msg.baselink_yaw
        self.baselink_pitch = msg.baselink_pitch
        self.baselink_roll = msg.baselink_roll
        self.publish_anomaly_map(anomaly_output["anomaly_map"], embedding_time, current_sonar_img)

    def detect_anomaly(self, embedding: np.ndarray) -> dict:
        """
        Computes an anomaly map from embeddings using nearest neighbor search.
        - Reshapes embeddings.
        - Finds nearest neighbors and anomaly scores.
        - Generates an anomaly map.
        """
        output_size = (224, 224)  # Desired output resolution
        embedding = self.reshape_embedding(embedding)

        # Compute nearest neighbor distances
        patch_scores, locations = self.nearest_neighbors(embedding, n_neighbors=1)

        # Reshape scores to match feature patch grid
        patch_scores = patch_scores.reshape((1, 1, 28, 28))

        # Generate the final anomaly map
        anomaly_map = self.anomaly_map_generator.compute_anomaly_map(patch_scores, output_size)

        return {"anomaly_map": anomaly_map}

    def publish_anomaly_map(self, anomaly_map: np.ndarray, timestamp: float, current_sonar_img):
        """
        Publishes the computed anomaly map as a ROS message.
        - Converts NumPy array to `Float32MultiArrayStamped` format.
        - Attaches the correct timestamp.
        """
        msg = Anomaly()
        msg.unix_time_ms = timestamp  # Preserve the original timestamp
        msg.data_image = current_sonar_img
        msg.data.data = anomaly_map.flatten().tolist()  # Flatten the anomaly map into a 1D list
        msg.source = 2
        msg.tracker_id = 0
        msg.sensor_setting_index = self.sonar_range
        msg.baselink_easting = self.baselink_easting
        msg.baselink_northing = self.baselink_northing
        msg.baselink_depth_msl = self.baselink_depth_msl
        msg.baselink_depth_surface = self.baselink_depth_surface
        msg.baselink_altitude = self.baselink_altitude
        msg.baselink_yaw = self.baselink_yaw
        msg.baselink_pitch = self.baselink_pitch
        msg.baselink_roll = self.baselink_roll
        self.anomaly_map_pub.publish(msg)  # Publish the message

    def nearest_neighbors(self, embedding: np.ndarray, n_neighbors: int) -> tuple:
        """
        Finds the nearest neighbors for the given embedding using Annoy.
        - Returns distances and indices of the nearest neighbors.
        """
        distances = []
        indices = []
        for vector in embedding:
            nearest_indices, nearest_distances = self.annoy_index.get_nns_by_vector(
                vector, n_neighbors, include_distances=True
            )
            distances.append(nearest_distances[0])
            indices.append(nearest_indices[0])
        return np.array(distances), np.array(indices)

    def reshape_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Reshapes the embedding from (batch, channels, height, width) to a flat feature vector."""
        embedding_size = embedding.shape[1]
        return embedding.transpose(0, 2, 3, 1).reshape(-1, embedding_size)


if __name__ == "__main__":
    rospy.init_node("anomaly_detection_node")

    memory_bank_path = rospy.get_param("~model")
    model = AnomalyDetector(memory_bank_path=memory_bank_path)

    rospy.spin()  # Keep the ROS node running

#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import math

from bx_msgs.msg import DetectedInstance, DetectedInstances, Anomaly, SurveyorInfo


class AnomalyInferencer:
    """
    ROS Node for Anomaly Inference.
    - Receives anomaly maps and sonar images.
    - Matches images with anomaly maps based on timestamps.
    - Publishes processed anomaly maps with visual overlays.
    """

    def __init__(self, normalize: bool = False):
        self.normalize = normalize  # Whether to normalize anomaly maps

        # **ROS Subscribers** (Listening for incoming messages)
        self.anomaly_map_sub = rospy.Subscriber(
            "/anomaly/detection_map", Anomaly, self.anomaly_map_callback, queue_size=1
        )
        # self.fls_data_sub = rospy.Subscriber("/ikan/fls/data", SurveyorInfo, self.fls_data_callback, queue_size=1)

        # **ROS Publishers** (Publishing processed images)
        self.compress_pub = rospy.Publisher("/anomaly/image_2/compressed", CompressedImage, queue_size=1)
        self.detection_pub = rospy.Publisher("/ikan/vision/ml/detections", DetectedInstances, queue_size=1)

        # **Placeholder for the latest sonar image**
        # self.polar_shape = (373, 256)  # Updated when a new image is received
        self.polar_shape = (224, 224)  # Updated when a new image is received


        # **Anomaly detection parameters for monopile with manta mine + oildrum**
        # self.threshold_anomaly = 45
        # self.threshold_visualiser = 0.68
        # self.threshold_centring = 0.5

        # **Anomaly detection parameters for blyth**

        self.threshold_anomaly = 50.0
        self.threshold_visualiser = 0.35

        self.threshold_centreing = 0.5


        # # **Anomaly detection parameters for oildrum**
        # self.threshold_anomaly = 44
        # self.threshold_visualiser = 0.68
        # self.threshold_centring = 0.5

        # **ROS message formats for detections**
        self.DetectedInstances = DetectedInstances()
        self.DetectedInstances.detector_name = "sim_fls_anomaly"

        # **Timestamp of the latest anomaly map received**
        self.current_time = rospy.Time()

        self.baselink_easting = 0
        self.baselink_northing = 0
        self.baselink_depth_msl = 0
        self.baselink_depth_surface = 0
        self.baselink_altitude = 0
        self.baselink_yaw = 0
        self.baselink_pitch = 0
        self.baselink_roll = 0

    def fls_data_callback(self, msg):
        self.range = msg.max_range
        self.fov_min = msg.fov_min
        self.fov_max = msg.fov_max

    def _normalize(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Normalize the anomaly map.
        - Applies centering and scaling.
        """
        results = (anomaly_map - self.threshold_anomaly) / (
            anomaly_map.max() - anomaly_map.min()
        ) + self.threshold_centreing
        return results

    def postprocess(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Resize the anomaly map to match the sonar image dimensions.
        - Uses bilinear interpolation.
        """
        anomaly_map = self._normalize(anomaly_map)
        return cv2.resize(anomaly_map, (self.polar_shape[1], self.polar_shape[0]), interpolation=cv2.INTER_LINEAR)

    def visualize(self, anomaly_map: np.ndarray, current_sonar_img) -> np.ndarray:
        """
        Overlay the anomaly map on the sonar image.
        - Finds anomaly regions using contours.
        - Draws detected regions and their centroids.
        """
        pred_score = anomaly_map.max()
        print(pred_score)
        segmented_map = (anomaly_map >= self.threshold_visualiser).astype(np.uint8) * anomaly_map
        segmented_overlay = (segmented_map * 255).astype(np.uint8)
        anomaly_contours, _ = cv2.findContours(segmented_overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Clear previous detections
        self.DetectedInstances.detections = []

        # Draw detected anomaly contours
        for contour in anomaly_contours:
            detected_instance = DetectedInstance()
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(segmented_overlay, (cX, cY), 2, (0, 0, 255), -1)
            else:
                continue  # Skip if contour is invalid

            # Compute the bounding box of the contour
            # cv2.rectangle(segmented_overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Convert contour data to ROS message format
            flatten_contour = contour.reshape(-1).astype(np.uint16).tolist()
            detected_instance.name = "anomaly"
            detected_instance.source = 130
            detected_instance.tracker_id = 0
            detected_instance.sensor_setting_index = self.sonar_range
            detected_instance.unix_time_ms = self.current_time
            detected_instance.baselink_easting = self.baselink_easting
            detected_instance.baselink_northing = self.baselink_northing
            detected_instance.baselink_depth_surface = self.baselink_depth_surface
            detected_instance.baselink_altitude = self.baselink_altitude
            detected_instance.baselink_yaw = self.baselink_yaw
            detected_instance.baselink_pitch = self.baselink_pitch
            detected_instance.baselink_roll = self.baselink_roll
            detected_instance.bbox_centre_x = cX
            detected_instance.bbox_centre_y = cY
            detected_instance.image_width = segmented_overlay.shape[1]
            detected_instance.image_height = segmented_overlay.shape[0]
            detected_instance.contour = flatten_contour
            detected_instance.confidence = int(round(pred_score * 100))

            # titre = (detected_instance.bbox_centre_x / detected_instance.image_width * (self.fov_max - self.fov_min) + self.fov_min) / 180.0 * math.pi
            # anomaly_range = detected_instance.bbox_centre_y / detected_instance.image_height * self.range
            # cartesian_x = anomaly_range * math.sin(titre)
            # cartesian_y = anomaly_range * math.cos(titre)
            # print("titre: ", titre, "anomaly range", anomaly_range)
            # print("x: ", cartesian_x, "y: ", cartesian_y)

            self.DetectedInstances.detections.append(detected_instance)

        self.detection_pub.publish(self.DetectedInstances)  # Publish detections


        print("threshold visual: ", self.threshold_visualiser)

        # Convert to heatmap format
        heatmap = cv2.cvtColor(segmented_overlay, cv2.COLOR_GRAY2BGR)

        # Blend sonar image and heatmap
        polar_resized = cv2.resize(
            current_sonar_img, (self.polar_shape[1], self.polar_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        blended = cv2.addWeighted(polar_resized, 0.8, heatmap, 0.5, 0)

        return blended

    def anomaly_map_callback(self, msg: Anomaly):
        """Receive anomaly map and process it."""

        # self.current_time = msg.header.stamp  # Store anomaly map timestamp
        self.current_time = msg.unix_time_ms  # Store anomaly map timestamp
        current_sonar_img = cv2.imdecode(np.frombuffer(msg.data_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        # current_sonar_img = msg.data_image
        anomaly_map_np = np.array(msg.data.data).reshape((224, 224))  # Ensure it's 2D before resizing
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
        self.infer(anomaly_map_np, current_sonar_img)

    def publish_compressed_image(self, image: np.ndarray):
        """Publish a compressed image."""
        _, compressed_image = cv2.imencode(".jpg", image)
        # print(image.shape)
        msg = CompressedImage()
        # msg.header.stamp = rospy.time(self.current_time)
        msg.header.stamp = rospy.Time.from_sec(self.current_time / 1000.0)
        msg.format = "jpeg"
        msg.data = np.array(compressed_image).tobytes()
        self.compress_pub.publish(msg)
        print("published")

    def infer(self, anomaly_map: np.ndarray, current_sonar_img) -> None:
        """Process the anomaly map and publish the final overlay image."""
        processed_map = self.postprocess(anomaly_map)

        overlay_image = self.visualize(processed_map, current_sonar_img)
        print("visualised")

        self.publish_compressed_image(overlay_image)


if __name__ == "__main__":
    rospy.init_node("Anomaly_inferencer_node")
    rospy.loginfo("Anomaly Inferencer Node Started")

    inferencer = AnomalyInferencer()
    rospy.spin()

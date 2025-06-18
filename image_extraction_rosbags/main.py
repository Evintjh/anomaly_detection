import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import rosbag
import glob
import argparse
import cv2
import csv
from utils.cos_similarity_filter import CS_filter


class Data_Prep:
    def __init__(self):
        self.CS_filter = CS_filter()  # Kept for compatibility, though not used here

    def extract_and_filter_frames_from_bag(
        self,
        bag_path,
        output_dir,
        use_sonar_as_primary,
        topic_name1,
        topic_name2=None,
        topic_name3=None,
        topic_name4=None,
        start_time=None,
        end_time=None,
        start_frame_index=0,
        threshold=0.98,
        batch_size=8,
    ):
        """Extract frames, filter with cosine similarity, and save unique results with aux data by copying original decoded images."""
        frame_count = start_frame_index

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize ResNet50 for feature extraction
        self.CS_filter = CS_filter()
        model = self.CS_filter.load_model()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
            ]
        )

        # Buffer for images and metadata
        primary_images_tensor = []  # Primary images as tensors for feature extraction
        primary_images_bgr = []  # Primary images as BGR numpy arrays for saving
        secondary_images = []  # Tagged images as BGR numpy arrays (from cv2.imdecode)
        metadata = []  # [timestamp, altitude, sonar_range/camera_tilt]
        file_names = []  # Primary file names

        bag = rosbag.Bag(bag_path, "r")
        bag_start_time = bag.get_start_time()
        abs_start_time = bag_start_time + start_time if start_time else bag_start_time
        abs_end_time = bag_start_time + end_time if end_time else bag.get_end_time()

        latest_camera_msg = None
        latest_sonar_msg = None
        latest_altitude_msg = None
        camera_tilt_angle = None

        print(f"Processing bag: {bag_path}")

        if use_sonar_as_primary:
            for topic, msg, t in bag.read_messages(topics=[topic_name1, topic_name2, topic_name3, topic_name4]):
                if t.to_sec() < abs_start_time:
                    continue
                if t.to_sec() > abs_end_time:
                    break
                timestamp = t.to_sec()

                if topic == topic_name3:  # Altitude topic
                    latest_altitude_msg = msg.pose.pose.position.z
                    continue
                if topic == topic_name4:  # Camera tilt topic
                    camera_tilt_angle = msg.data
                    continue
                if topic == topic_name2:  # Camera topic
                    latest_camera_msg = msg
                    continue
                if topic == topic_name1 and latest_camera_msg:  # Sonar topic with available camera frame
                    try:
                        # Processing and collecting images
                        sonar_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                        if sonar_image is None:
                            print(
                                f"Failed to decode sonar image at {t.to_sec()}: data length = {len(msg.data)}, format = {getattr(msg, 'format', 'N/A')}"
                            )
                            continue

                        # sonar_range = round(msg.max_range)
                        camera_image = cv2.imdecode(np.frombuffer(latest_camera_msg.data, np.uint8), cv2.IMREAD_COLOR)
                        if camera_image is None:
                            print(
                                f"Failed to decode camera image at {t.to_sec()}: data length = {len(latest_camera_msg.data)}, format = {getattr(latest_camera_msg, 'format', 'N/A')}"
                            )
                            continue

                        # Convert to PIL for feature extraction (RGB)
                        sonar_pil = Image.fromarray(cv2.cvtColor(sonar_image, cv2.COLOR_BGR2RGB))
                        sonar_tensor = transform(sonar_pil)
                        primary_images_tensor.append(sonar_tensor)
                        primary_images_bgr.append(sonar_image.copy())  # Store BGR numpy array for saving
                        secondary_images.append(camera_image.copy())  # Store BGR camera image for tagged folder
                        file_names.append(f"sonar_{frame_count:06d}_{round(timestamp*1000)}.jpg")
                        # metadata.append([timestamp, latest_altitude_msg, sonar_range])
                        metadata.append([timestamp, latest_altitude_msg])

                        frame_count += 1
                    except Exception as e:
                        print(f"Error processing sonar or camera frame: {e}")
                        continue
        else:
            for topic, msg, t in bag.read_messages(topics=[topic_name1, topic_name2, topic_name3, topic_name4]):
                if t.to_sec() < abs_start_time:
                    continue
                if t.to_sec() > abs_end_time:
                    break
                timestamp = t.to_sec()

                if topic == topic_name3:  # Altitude topic
                    latest_altitude_msg = msg.pose.pose.position.z
                    continue
                if topic == topic_name4:  # Camera tilt topic
                    camera_tilt_angle = msg.data
                    continue
                if topic == topic_name1:  # Sonar topic
                    latest_sonar_msg = msg
                    continue
                if topic == topic_name2 and latest_sonar_msg:  # Camera topic with available sonar data
                    try:
                        # Processing and collecting images
                        camera_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                        if camera_image is None:
                            print(
                                f"Failed to decode camera image at {t.to_sec()}: data length = {len(msg.data)}, format = {getattr(msg, 'format', 'N/A')}"
                            )
                            continue

                        sonar_image = cv2.imdecode(np.frombuffer(latest_sonar_msg.data, np.uint8), cv2.IMREAD_COLOR)
                        if sonar_image is None:
                            print(
                                f"Failed to decode sonar image at {t.to_sec()}: data length = {len(latest_sonar_msg.data)}, format = {getattr(latest_sonar_msg, 'format', 'N/A')}"
                            )
                            continue

                        # Convert to PIL for feature extraction (RGB)
                        camera_pil = Image.fromarray(cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB))
                        camera_tensor = transform(camera_pil)
                        primary_images_tensor.append(camera_tensor)
                        primary_images_bgr.append(camera_image.copy())  # Store BGR numpy array for saving
                        secondary_images.append(sonar_image.copy())  # Store BGR sonar image for tagged folder
                        file_names.append(f"front_{frame_count:06d}_{round(timestamp * 1000)}.jpg")
                        metadata.append([timestamp, latest_altitude_msg, camera_tilt_angle])

                        frame_count += 1
                    except Exception as e:
                        print(f"Error processing camera or sonar frame: {e}")
                        continue

        bag.close()
        print(f"Extracted {len(primary_images_tensor)} frames from {bag_path}")

        if not primary_images_tensor:
            print("No images to filter.")
            return frame_count

        # Collating embeddings
        images = torch.stack(primary_images_tensor)
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].cuda()
            with torch.no_grad():
                batch_embeddings = model(batch).squeeze(-1).squeeze(-1)
            embeddings.append(batch_embeddings.cpu())
        embeddings = torch.cat(embeddings).numpy()

        # Compute cosine similarity using sklearn
        embeddings = np.array(embeddings)  # Ensure embeddings are in NumPy format
        unique_indices = [0]  # Start with the first image as unique
        selected_features = [embeddings[0]]  # Store the first embedding

        unique_indices = self.CS_filter.filter_unique_files(selected_features, embeddings, threshold)

        # Save filtered results by copying original decoded images
        csv_file = os.path.join(output_dir, "aux_data.csv")
        csv_header = (
            ["file_name", "unix_time_ms", "image_id", "baselink_altitude", "sonar_range"]
            if use_sonar_as_primary
            else ["file_name", "unix_time_ms", "image_id", "baselink_altitude", "camera_tilt"]
        )
        csv_data = []

        related_images_dir = os.path.join(output_dir, "related_images")
        os.makedirs(related_images_dir, exist_ok=True)  # Ensure the directory exists
        for idx in unique_indices:
            primary_file = file_names[idx]
            primary_base = os.path.splitext(primary_file)[0]
            tagged_folder = primary_base + "_jpg"

            # Save primary image by copying the decoded BGR image
            primary_full_path = os.path.join(output_dir, primary_file)
            primary_image = primary_images_bgr[idx].copy()  # Use the original BGR numpy array from primary decoding
            cv2.imwrite(primary_full_path, primary_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # High quality

            prefix_length = len("sonar") if use_sonar_as_primary else len("front")

            # Save tagged folder by copying the decoded BGR image (secondary)
            os.makedirs(os.path.join(related_images_dir, tagged_folder), exist_ok=True)
            # Extract correct substring dynamically
            tagged_subfilename = os.path.join(
                related_images_dir,
                tagged_folder,
                (
                    f"front{primary_base[prefix_length:]}.jpg"
                    if use_sonar_as_primary
                    else f"sonar{primary_base[prefix_length:]}.jpg"
                ),
            )
            cv2.imwrite(tagged_subfilename, secondary_images[idx], [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # High quality

        #     # Add CSV row
        #     # timestamp, altitude, extra = metadata[idx]
        #     timestamp, altitude = metadata[idx]

        #     if altitude is None:
        #         altitude = float("nan")
        #     extra = extra if extra is not None else float("nan")
        #     csv_row = [
        #         primary_file,
        #         round(timestamp * 1000),
        #         f"{idx:06d}",  # Use filtered index
        #         altitude,
        #         extra,
        #     ]
        #     csv_data.append(csv_row)
        #     print(f"Saved {primary_file} and tagged folder {tagged_folder}")

        # with open(csv_file, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(csv_header)
        #     writer.writerows(csv_data)

        # print(f"Filtered and saved {len(unique_indices)} unique images from {len(file_names)} total.")
        # with open(csv_file, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(csv_header)
        #     writer.writerows(csv_data)

        # # Post-process CSV: Sort by image_id after writing
        # if os.path.exists(csv_file):
        #     with open(csv_file, "r", newline="") as f:
        #         reader = csv.reader(f)
        #         csv_data_sorted = list(reader)

        #     # Extract headers and data separately
        #     csv_header = csv_data_sorted[0]  # First row is header
        #     csv_data_sorted = csv_data_sorted[1:]  # Remaining rows are data

        #     # Sort by image_id (column index 2, which is `image_id`)
        #     csv_data_sorted.sort(key=lambda row: int(row[2]))  # Convert image_id to int for proper sorting

        #     # Write sorted data back to CSV
        #     with open(csv_file, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(csv_header)
        #         writer.writerows(csv_data_sorted)

        #     print(f"CSV sorted by image_id and updated successfully: {csv_file}")

        return frame_count

    def extract_and_filter_from_all_bags(
        self,
        directory,
        output_dir,
        use_sonar_as_primary,
        topic_name1,
        topic_name2=None,
        topic_name3=None,
        topic_name4=None,
        start_time=None,
        end_time=None,
        start_frame_index=0,
        threshold=0.98,
        batch_size=8,
    ):
        rosbag_files = sorted(glob.glob(os.path.join(directory, "*.beex")))
        if not rosbag_files:
            print("No rosbag files found in the directory!")
            return
        print(f"Found {len(rosbag_files)} rosbag files. Processing...")
        frame_count = start_frame_index
        for bag_path in rosbag_files:
            frame_count = self.extract_and_filter_frames_from_bag(
                bag_path,
                output_dir,
                use_sonar_as_primary,
                topic_name1,
                topic_name2,
                topic_name3,
                topic_name4,
                start_time,
                end_time,
                frame_count,
                threshold,
                batch_size,
            )


if __name__ == "__main__":

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    parser = argparse.ArgumentParser(description="Extract and filter frames from rosbags")
    parser.add_argument(
        "--use_sonar_as_primary",
        type=str_to_bool,
        default=False,
        help="True: sonar drives sync, False: camera drives sync",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/mnt/d/FullData/Normal_seabed_with_no_anomaly/tuas pipeline aug 2024/POLAR/20_Feb_2025_test/2025-02-20/good_bags/8m_range_ONLY_ACCURATE_TESTING",
        help="Folder containing rosbag files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/mnt/d/FullData/Normal_seabed_with_no_anomaly/tuas pipeline aug 2024/POLAR/20_Feb_2025_test/2025-02-20/good_bags/8m_range_ONLY_ACCURATE_TESTING/testing_filtered_final",
        help="Folder for filtered unique images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Cosine similarity threshold for duplicates",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for cosine similarity filtering",
    )

    args = parser.parse_args()

    rosbag_directory = args.input_folder
    output_directory = args.output_folder
    topic_name1 = "/ikan/sonar/image/compressed"
    topic_name2 = "/ikan/front_cam/ml_clahe/compressed"
    topic_name3 = "/ikan/ins/altitude/filtered"
    topic_name4 = "/ikan/hardware/cameraTilt/angle"
    start_time = 280
    end_time = None
    start_frame_index = 0

    data_prep = Data_Prep()
    data_prep.extract_and_filter_from_all_bags(
        rosbag_directory,
        output_directory,
        args.use_sonar_as_primary,
        topic_name1,
        topic_name2,
        topic_name3,
        topic_name4,
        start_time,
        end_time,
        start_frame_index,
        args.threshold,
        args.batch_size,
    )

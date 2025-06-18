import cv2
import os
import numpy as np

def crop_and_save_images(input_folder, output_folder, crop_size=398):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Supported image formats
            # Construct full file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load image {input_path}")
                continue

            # Get image dimensions
            height, width = image.shape[:2]

            # Calculate the center of the image
            center_x = width // 2
            center_y = height // 2

            # Calculate the starting coordinates for the crop
            # Ensure the crop stays within the image boundaries
            start_x = max(0, center_x - crop_size // 2)
            start_y = max(0, center_y - crop_size // 2)
            end_x = min(width, start_x + crop_size)
            end_y = min(height, start_y + crop_size)

            # Adjust start coordinates if the crop would exceed image boundaries
            if end_x - start_x < crop_size:
                start_x = max(0, width - crop_size)
            if end_y - start_y < crop_size:
                start_y = max(0, height - crop_size)

            # Define the ROI (Region of Interest) for cropping
            roi = image[start_y:end_y, start_x:end_x]

            # Resize or pad if the cropped region is smaller than 398x398
            if roi.shape[0] < crop_size or roi.shape[1] < crop_size:
                # Create a blank image of 398x398
                cropped_image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                # Calculate padding offsets
                y_offset = (crop_size - roi.shape[0]) // 2
                x_offset = (crop_size - roi.shape[1]) // 2
                # Place the cropped region in the center of the blank image
                cropped_image[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi
            else:
                cropped_image = roi

            # Ensure the output is exactly 398x398 (resize if necessary)
            if cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
                cropped_image = cv2.resize(cropped_image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

            # Save the cropped image
            cv2.imwrite(output_path, cropped_image)
            print(f"Saved cropped image to {output_path}")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "/mnt/d/FullData/Normal_seabed_with_no_anomaly/blyth/cartesian/normal_again_no_aug_PROPER"  # Replace with your input folder path
    output_folder = "/mnt/d/FullData/Normal_seabed_with_no_anomaly/blyth/cartesian/final_normal_again_no_aug_PROPER"  # Replace with your output folder path

    # Crop and save images
    crop_and_save_images(input_folder, output_folder)


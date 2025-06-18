#!/home/ashwin/anaconda3/envs/anamoly_detection/bin/python3
import rospy
import torch
import numpy as np
import rospkg
import os


class AnomalyInferencer:
    def __init__(
        self, model_path: str, model_root_path: str, device: str = "auto", normalize: bool = False, filter_size: int = 1
    ):
        self.model_root_path = model_root_path
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.normalize = normalize

    @staticmethod
    def _get_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)

    def _load_model(self, path: str) -> torch.nn.Module:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model = checkpoint["model"]

        # Convert the PyTorch tensor to a NumPy array
        memory_bank_np = model.model.memory_bank.cpu().numpy()  # Ensure it's on CPU

        # Save it as a .npy file
        memory_bank_path = os.path.join(self.model_root_path, "final_blyth_model.npy")
        np.save(memory_bank_path, memory_bank_np)
        print("Memory bank saved as 'final_blyth_model.npy'")

        # torch.save(model.model.memory_bank, "model_610_cos_memory_bank.pt")
        # raise "CHECKPT"
        model.eval()
        return model.to(self.device)


if __name__ == "__main__":

    # Initialize ROS node
    rospy.init_node("enhanced_inferencer_node")

    # Use rospack to find the model path dynamically
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("anomaly_inference")
    model_root_dir = f"{package_path}/models"
    model_path = f"{package_path}/models/final_blyth_model.pt"

    device = rospy.get_param("~device", "cpu")  # Provide default device as "auto" if not set
    filter_size = rospy.get_param("~filter_size", 1)  # Default filter size is 1
    rospy.loginfo(f"Using model weights: {model_path}")
    rospy.loginfo(f"Running on device: {device}")
    # Create the inferencer instance
    inferencer = AnomalyInferencer(
        model_path=model_path, model_root_path=model_root_dir, device=device, filter_size=filter_size
    )

    # Keep the node running
    rospy.spin()

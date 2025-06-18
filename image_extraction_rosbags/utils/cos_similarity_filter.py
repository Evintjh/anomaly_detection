import os
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np


class CS_filter:
    def __init__(self):
        pass

    # Function to load the ResNet18 model pre-trained on ImageNet
    def load_model(self):
        # Initialize ResNet50 for feature extraction
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
        model.cuda().eval()
        return model

    # Function to extract features from an image using ResNet18
    def extract_features(self, model, image, transform):
        # image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # No need to compute gradients
            features = model(image.cuda())
        return features.cpu().squeeze().numpy()  # Remove the batch dimension and return numpy array

    # Function to calculate cosine similarity
    def calculate_cosine_similarity(self, features1, features2):
        return cosine_similarity([features1], [features2])

    def filter_unique_files(self, baseline_features, embeddings, threshold):
        unique_indices = [0]  # Start with the first image as unique
        tqdm_bar = tqdm(range(1, len(embeddings)), desc="Filtering unique images")
        for i in tqdm_bar:
            query_feature = embeddings[i].reshape(1, -1)  # Query feature
            similarities = cosine_similarity(query_feature, baseline_features)  # Compare with existing features

            if np.max(similarities) < threshold:  # If no match exceeds threshold
                baseline_features.append(embeddings[i])
                unique_indices.append(i)

            tqdm_bar.set_description(
                f"Unique: {len(unique_indices)} | Similarity: Min {np.min(similarities):.2f} Max {np.max(similarities):.2f}"
            )

        return unique_indices

"""
ArcFace Implementation - Additive Angular Margin Loss for Face Recognition
Research Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import math


class ArcMarginProduct(nn.Module):
    """
    ArcFace margin product layer
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Cosine of theta
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class BackboneNetwork(nn.Module):
    """
    ResNet-based backbone for ArcFace
    """

    def __init__(self, backbone='resnet50', pretrained=True):
        super(BackboneNetwork, self).__init__()

        if backbone == 'resnet50':
            from torchvision.models import resnet50
            base_model = resnet50(pretrained=pretrained)

            # Remove the last fully connected layer
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.in_features = base_model.fc.in_features

        elif backbone == 'resnet101':
            from torchvision.models import resnet101
            base_model = resnet101(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.in_features = base_model.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class ArcFaceModel:
    """
    ArcFace Model Implementation with research features
    """

    def __init__(self, backbone='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.backbone_type = backbone

        # Create backbone
        self.backbone = BackboneNetwork(backbone, pretrained=True).to(device)

        # Create ArcFace head
        self.embedding_dim = 512  # Standard ArcFace embedding dimension
        self.arcface_head = ArcMarginProduct(
            in_features=self.backbone.in_features,
            out_features=1000,  # Large number for research
            s=30.0,
            m=0.5
        ).to(device)

        # Final embedding layer
        self.embedding_layer = nn.Linear(self.backbone.in_features, self.embedding_dim).to(device)

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),  # ArcFace standard input size
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Set to eval mode
        self.backbone.eval()
        self.arcface_head.eval()
        self.embedding_layer.eval()

        print(f"✅ ArcFace model loaded ({backbone} backbone) on {device}")

    def extract_embeddings(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract ArcFace embeddings
        """
        if not face_images:
            return np.array([])

        embeddings = []

        for img in face_images:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess
            img_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.backbone(img_tensor)
                embedding = self.embedding_layer(features)
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy().flatten()

            embeddings.append(embedding)

        return np.array(embeddings)

    def extract_features(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep features before ArcFace margin (for research analysis)
        """
        features_list = []

        for img in face_images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.backbone(img_tensor)
                features = features.cpu().numpy().flatten()

            features_list.append(features)

        return np.array(features_list)

    def compute_angular_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute angular similarity (ArcFace-specific)
        """
        # Normalize embeddings
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)

        # Cosine similarity
        cos_theta = np.dot(e1, e2)

        # Clip to valid range
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Angular distance
        theta = np.arccos(cos_theta)
        angular_similarity = 1.0 - (theta / np.pi)  # Convert to similarity [0, 1]

        return float(angular_similarity)

    def verify_faces(self, embedding1: np.ndarray, embedding2: np.ndarray,
                     threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verify faces using ArcFace angular similarity
        """
        similarity = self.compute_angular_similarity(embedding1, embedding2)
        is_same = similarity > threshold

        return is_same, similarity

    def identify_face(self, query_embedding: np.ndarray,
                      database_embeddings: np.ndarray,
                      database_labels: List[str],
                      threshold: float = 0.5) -> Tuple[str, float, bool]:
        """
        Identify face using ArcFace
        """
        if len(database_embeddings) == 0:
            return "Unknown", 0.0, False

        # Compute angular similarities
        similarities = []
        for db_embedding in database_embeddings:
            similarity = self.compute_angular_similarity(query_embedding, db_embedding)
            similarities.append(similarity)

        # Find best match
        max_similarity = max(similarities)
        best_index = np.argmax(similarities)

        if max_similarity > threshold:
            return database_labels[best_index], max_similarity, True
        else:
            return "Unknown", max_similarity, False

    def batch_process(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Batch processing for efficiency
        """
        if not face_images:
            return np.array([])

        # Preprocess batch
        batch_tensors = []
        for img in face_images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.preprocess(img_rgb)
            batch_tensors.append(img_tensor)

        batch = torch.stack(batch_tensors).to(self.device)

        # Extract embeddings in batch
        with torch.no_grad():
            features = self.backbone(batch)
            embeddings = self.embedding_layer(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def train_with_arcface_loss(self, train_loader, num_classes, epochs=20, lr=0.001):
        """
        Train model with ArcFace loss (for research)
        """
        print(f"🔧 Training ArcFace model for {epochs} epochs...")

        # Switch to train mode
        self.backbone.train()
        self.arcface_head.train()

        # Create ArcFace head for training
        arcface = ArcMarginProduct(
            in_features=self.backbone.in_features,
            out_features=num_classes,
            s=30.0,
            m=0.5
        ).to(self.device)

        # Classifier
        classifier = nn.Linear(self.backbone.in_features, num_classes).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': arcface.parameters()},
            {'params': classifier.parameters()}
        ], lr=lr)

        # Loss functions
        arcface_loss_fn = nn.CrossEntropyLoss()
        classifier_loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            total_loss = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                features = self.backbone(images)

                # ArcFace loss
                arcface_output = arcface(features, labels)
                arcface_loss = arcface_loss_fn(arcface_output, labels)

                # Classifier loss
                classifier_output = classifier(features)
                classifier_loss = classifier_loss_fn(classifier_output, labels)

                # Total loss
                loss = arcface_loss + classifier_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Switch back to eval mode
        self.backbone.eval()
        arcface.eval()
        classifier.eval()

        print("✅ ArcFace training completed")

        # Update embedding layer with trained weights
        self.embedding_layer = classifier

        return arcface, classifier

    def get_model_info(self) -> Dict:
        """
        Get model information
        """
        return {
            'model_name': 'ArcFace',
            'backbone': self.backbone_type,
            'embedding_dim': self.embedding_dim,
            'margin': 0.5,
            'scale': 30.0,
            'input_size': (112, 112),
            'paper': 'Deng et al., 2019. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"',
            'device': str(self.device),
            'features': {
                'angular_margin': True,
                'normalized_embeddings': True,
                'research_ready': True
            }
        }

    def analyze_embedding_space(self, embeddings: np.ndarray, labels: List[str]):
        """
        Analyze embedding space for research insights
        """
        print("📊 Analyzing ArcFace embedding space...")

        # Compute intra-class and inter-class distances
        unique_labels = list(set(labels))

        intra_class_distances = []
        inter_class_distances = []

        for label in unique_labels:
            # Get embeddings for this class
            class_indices = [i for i, l in enumerate(labels) if l == label]
            class_embeddings = embeddings[class_indices]

            if len(class_embeddings) > 1:
                # Intra-class distances
                for i in range(len(class_embeddings)):
                    for j in range(i + 1, len(class_embeddings)):
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                        intra_class_distances.append(dist)

        # Inter-class distances
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i + 1:], i + 1):
                emb1 = embeddings[labels.index(label1)]
                emb2 = embeddings[labels.index(label2)]
                dist = np.linalg.norm(emb1 - emb2)
                inter_class_distances.append(dist)

        # Compute statistics
        intra_mean = np.mean(intra_class_distances) if intra_class_distances else 0
        intra_std = np.std(intra_class_distances) if intra_class_distances else 0
        inter_mean = np.mean(inter_class_distances) if inter_class_distances else 0
        inter_std = np.std(inter_class_distances) if inter_class_distances else 0

        # Separation ratio (higher is better)
        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')

        return {
            'intra_class_mean': float(intra_mean),
            'intra_class_std': float(intra_std),
            'inter_class_mean': float(inter_mean),
            'inter_class_std': float(inter_std),
            'separation_ratio': float(separation_ratio),
            'num_classes': len(unique_labels),
            'total_samples': len(embeddings)
        }
"""
Attention-based Face Recognition with Spatial and Channel Attention
Research-level implementation with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import math


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module
    Focuses on important facial regions
    """

    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)

        # Apply attention
        return x * attention


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation)
    Emphasizes important feature channels
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)

        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)

        # Combine
        attention = avg_out + max_out

        return x * attention.expand_as(x)


class DualAttentionBlock(nn.Module):
    """
    Dual Attention Block with Spatial and Channel Attention
    """

    def __init__(self, in_channels):
        super(DualAttentionBlock, self).__init__()

        self.channel_attention = ChannelAttentionModule(in_channels)
        self.spatial_attention = SpatialAttentionModule(in_channels)

    def forward(self, x):
        # Channel attention first
        x = self.channel_attention(x)

        # Then spatial attention
        x = self.spatial_attention(x)

        return x


class AttentionBackbone(nn.Module):
    """
    Attention-based Backbone Network
    """

    def __init__(self, base_backbone='resnet50'):
        super(AttentionBackbone, self).__init__()

        if base_backbone == 'resnet50':
            from torchvision.models import resnet50
            base_model = resnet50(pretrained=True)

            # Extract layers
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool

            # Layer 1 with attention
            self.layer1 = base_model.layer1
            self.attention1 = DualAttentionBlock(256)

            # Layer 2 with attention
            self.layer2 = base_model.layer2
            self.attention2 = DualAttentionBlock(512)

            # Layer 3 with attention
            self.layer3 = base_model.layer3
            self.attention3 = DualAttentionBlock(1024)

            # Layer 4 with attention
            self.layer4 = base_model.layer4
            self.attention4 = DualAttentionBlock(2048)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.in_features = 2048

        elif base_backbone == 'resnet101':
            from torchvision.models import resnet101
            base_model = resnet101(pretrained=True)

            # Similar structure as above
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool

            self.layer1 = base_model.layer1
            self.attention1 = DualAttentionBlock(256)

            self.layer2 = base_model.layer2
            self.attention2 = DualAttentionBlock(512)

            self.layer3 = base_model.layer3
            self.attention3 = DualAttentionBlock(1024)

            self.layer4 = base_model.layer4
            self.attention4 = DualAttentionBlock(2048)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.in_features = 2048

        else:
            raise ValueError(f"Unsupported backbone: {base_backbone}")

    def forward(self, x, return_attention_maps=False):
        attention_maps = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 with attention
        x = self.layer1(x)
        x = self.attention1(x)
        if return_attention_maps:
            attention_maps['layer1'] = x

        # Layer 2 with attention
        x = self.layer2(x)
        x = self.attention2(x)
        if return_attention_maps:
            attention_maps['layer2'] = x

        # Layer 3 with attention
        x = self.layer3(x)
        x = self.attention3(x)
        if return_attention_maps:
            attention_maps['layer3'] = x

        # Layer 4 with attention
        x = self.layer4(x)
        x = self.attention4(x)
        if return_attention_maps:
            attention_maps['layer4'] = x

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if return_attention_maps:
            return x, attention_maps
        else:
            return x


class AttentionFaceModel:
    """
    Attention-based Face Recognition Model
    Research: "Attention-guided Face Recognition for Occluded and Challenging Conditions"
    """

    def __init__(self, backbone='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.backbone_type = backbone

        # Create attention backbone
        self.backbone = AttentionBackbone(backbone).to(device)

        # Embedding layer
        self.embedding_dim = 512
        self.embedding_layer = nn.Linear(self.backbone.in_features, self.embedding_dim).to(device)

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Set to eval mode
        self.backbone.eval()
        self.embedding_layer.eval()

        print(f"✅ AttentionFace model loaded ({backbone} with dual attention) on {device}")

    def extract_embeddings(self, face_images: List[np.ndarray],
                           return_attention: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract embeddings with optional attention maps
        """
        if not face_images:
            return np.array([]), None if return_attention else np.array([])

        embeddings = []
        all_attention_maps = [] if return_attention else None

        for img in face_images:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess
            img_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)

            # Extract features with attention
            with torch.no_grad():
                if return_attention:
                    features, attention_maps = self.backbone(img_tensor, return_attention_maps=True)
                    all_attention_maps.append(attention_maps)
                else:
                    features = self.backbone(img_tensor)

                # Get embedding
                embedding = self.embedding_layer(features)
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy().flatten()

            embeddings.append(embedding)

        embeddings_array = np.array(embeddings)

        if return_attention:
            return embeddings_array, all_attention_maps
        else:
            return embeddings_array

    def visualize_attention(self, face_image: np.ndarray, save_path: str = None):
        """
        Visualize attention maps for research analysis
        """
        import matplotlib.pyplot as plt

        # Extract embeddings with attention maps
        embeddings, attention_maps_list = self.extract_embeddings([face_image], return_attention=True)

        if not attention_maps_list:
            print("No attention maps available")
            return

        attention_maps = attention_maps_list[0]

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        # Original image
        img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Face')
        axes[0].axis('off')

        # Plot attention maps from each layer
        for idx, (layer_name, att_map) in enumerate(attention_maps.items(), 1):
            if idx >= len(axes):
                break

            # Average across channels
            avg_attention = att_map.mean(dim=1).squeeze().cpu().numpy()

            # Resize to match input size for visualization
            if avg_attention.ndim == 2:
                heatmap = cv2.resize(avg_attention, (face_image.shape[1], face_image.shape[0]))

                # Overlay on original image
                axes[idx].imshow(img_rgb)
                axes[idx].imshow(heatmap, cmap='jet', alpha=0.5)
                axes[idx].set_title(f'Attention: {layer_name}')
                axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Attention visualization saved to {save_path}")

        plt.show()

        return attention_maps

    def analyze_attention_patterns(self, face_images: List[np.ndarray], labels: List[str]):
        """
        Analyze attention patterns for different face attributes
        For research on what the model focuses on
        """
        print("🔍 Analyzing attention patterns...")

        # Extract embeddings with attention
        embeddings, attention_maps_list = self.extract_embeddings(face_images, return_attention=True)

        if not attention_maps_list:
            return {}

        # Analyze attention distribution
        attention_stats = {}

        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_attentions = []

            for att_maps in attention_maps_list:
                if layer in att_maps:
                    # Get attention map and compute statistics
                    att_map = att_maps[layer]

                    # Mean attention value
                    mean_attention = att_map.mean().item()

                    # Attention variance (how focused it is)
                    var_attention = att_map.var().item()

                    # Entropy of attention (higher = more dispersed)
                    att_flat = att_map.flatten()
                    att_probs = F.softmax(att_flat, dim=0)
                    entropy = -torch.sum(att_probs * torch.log(att_probs + 1e-10)).item()

                    layer_attentions.append({
                        'mean': mean_attention,
                        'variance': var_attention,
                        'entropy': entropy
                    })

            if layer_attentions:
                # Aggregate statistics
                attention_stats[layer] = {
                    'mean_attention': np.mean([a['mean'] for a in layer_attentions]),
                    'mean_variance': np.mean([a['variance'] for a in layer_attentions]),
                    'mean_entropy': np.mean([a['entropy'] for a in layer_attentions]),
                    'num_samples': len(layer_attentions)
                }

        # Correlation with face recognition performance
        print("\n📈 Attention Statistics:")
        for layer, stats in attention_stats.items():
            print(f"\n{layer}:")
            print(f"  Mean Attention: {stats['mean_attention']:.4f}")
            print(f"  Attention Variance: {stats['mean_variance']:.4f}")
            print(f"  Attention Entropy: {stats['mean_entropy']:.4f}")

        return attention_stats

    def compute_attention_similarity(self, face1: np.ndarray, face2: np.ndarray) -> Dict[str, float]:
        """
        Compute similarity based on attention patterns
        Useful for analyzing if faces are attended to similarly
        """
        # Extract attention maps for both faces
        _, att_maps1 = self.extract_embeddings([face1], return_attention=True)
        _, att_maps2 = self.extract_embeddings([face2], return_attention=True)

        if not att_maps1 or not att_maps2:
            return {}

        att_maps1 = att_maps1[0]
        att_maps2 = att_maps2[0]

        similarities = {}

        for layer in att_maps1.keys():
            if layer in att_maps2:
                # Flatten attention maps
                att1 = att_maps1[layer].flatten().cpu().numpy()
                att2 = att_maps2[layer].flatten().cpu().numpy()

                # Compute cosine similarity
                similarity = np.dot(att1, att2) / (np.linalg.norm(att1) * np.linalg.norm(att2))
                similarities[layer] = float(similarity)

        return similarities

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        """
        return {
            'model_name': 'AttentionFace',
            'backbone': self.backbone_type,
            'embedding_dim': self.embedding_dim,
            'attention_modules': ['SpatialAttention', 'ChannelAttention', 'DualAttention'],
            'input_size': (224, 224),
            'research_contribution': 'Attention-guided feature learning for face recognition',
            'device': str(self.device),
            'features': {
                'attention_visualization': True,
                'attention_analysis': True,
                'pattern_similarity': True,
                'research_ready': True
            }
        }

    def occluded_face_recognition(self, occluded_faces: List[np.ndarray],
                                  occlusion_masks: List[np.ndarray] = None):
        """
        Test model performance on occluded faces
        For research on robustness
        """
        print("🧪 Testing occluded face recognition...")

        # Extract embeddings for occluded faces
        occluded_embeddings = self.extract_embeddings(occluded_faces)

        # If masks provided, analyze attention on occluded regions
        if occlusion_masks:
            attention_responses = []

            for face, mask in zip(occluded_faces, occlusion_masks):
                # Extract attention
                _, att_maps = self.extract_embeddings([face], return_attention=True)

                if att_maps:
                    # Analyze attention in occluded vs non-occluded regions
                    att_map = att_maps[0]['layer4']  # Final layer attention

                    # Resize mask to match attention map size
                    mask_resized = cv2.resize(mask.astype(float),
                                              (att_map.shape[3], att_map.shape[2]))

                    # Compute attention in occluded regions
                    occluded_attention = (att_map * mask_resized).sum().item()
                    total_attention = att_map.sum().item()

                    if total_attention > 0:
                        occlusion_ratio = occluded_attention / total_attention
                        attention_responses.append(occlusion_ratio)

            if attention_responses:
                print(f"📊 Average attention on occluded regions: {np.mean(attention_responses):.3f}")

        return {
            'num_occluded_faces': len(occluded_faces),
            'embeddings_shape': occluded_embeddings.shape if occluded_embeddings.size > 0 else None,
            'attention_analysis': bool(occlusion_masks)
        }
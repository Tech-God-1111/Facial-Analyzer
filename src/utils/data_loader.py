"""
Data loading utilities for face recognition research
Supports multiple datasets and preprocessing pipelines
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import random
from dataclasses import dataclass
from pathlib import Path
import pickle
import json


@dataclass
class FaceSample:
    """Data class for face samples"""
    image: np.ndarray
    label: str
    image_path: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    landmarks: Optional[List[Tuple[int, int]]] = None
    attributes: Optional[Dict] = None

    def __post_init__(self):
        """Validate and preprocess after initialization"""
        if self.image is not None:
            # Ensure image is uint8
            if self.image.dtype != np.uint8:
                self.image = (self.image * 255).astype(np.uint8)

            # Convert to RGB if needed
            if len(self.image.shape) == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            elif self.image.shape[2] == 4:
                self.image = self.image[:, :, :3]
            elif self.image.shape[2] == 1:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)


class FaceDataset:
    """
    Base class for face datasets
    Supports multiple dataset formats and preprocessing
    """

    def __init__(self, root_dir: str, name: str = "custom"):
        self.root_dir = Path(root_dir)
        self.name = name
        self.samples: List[FaceSample] = []
        self.label_to_indices: Dict[str, List[int]] = {}
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset - to be implemented by subclasses"""
        pass

    def load_images_from_folder(self,
                                folder_structure: str = "label_folders",
                                extensions: List[str] = None):
        """
        Load images from folder structure
        Args:
            folder_structure: "label_folders" or "flat_with_metadata"
            extensions: List of image extensions to load
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        self.samples = []
        self.label_to_indices = {}

        if folder_structure == "label_folders":
            # Structure: root/label/image1.jpg
            for label_dir in self.root_dir.iterdir():
                if label_dir.is_dir():
                    label = label_dir.name
                    label_indices = []

                    for img_file in label_dir.iterdir():
                        if img_file.suffix.lower() in extensions:
                            try:
                                image = cv2.imread(str(img_file))
                                if image is not None:
                                    sample = FaceSample(
                                        image=image,
                                        label=label,
                                        image_path=str(img_file)
                                    )
                                    idx = len(self.samples)
                                    self.samples.append(sample)
                                    label_indices.append(idx)
                            except Exception as e:
                                print(f"Error loading {img_file}: {e}")

                    if label_indices:
                        self.label_to_indices[label] = label_indices

        elif folder_structure == "flat_with_metadata":
            # Structure: root/images/ with metadata file
            images_dir = self.root_dir / "images"
            metadata_file = self.root_dir / "metadata.json"

            if images_dir.exists() and metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                for img_info in metadata:
                    img_path = images_dir / img_info['filename']
                    if img_path.exists():
                        try:
                            image = cv2.imread(str(img_path))
                            if image is not None:
                                sample = FaceSample(
                                    image=image,
                                    label=img_info['label'],
                                    image_path=str(img_path),
                                    bbox=img_info.get('bbox'),
                                    landmarks=img_info.get('landmarks'),
                                    attributes=img_info.get('attributes')
                                )
                                idx = len(self.samples)
                                self.samples.append(sample)

                                # Update label indices
                                label = img_info['label']
                                if label not in self.label_to_indices:
                                    self.label_to_indices[label] = []
                                self.label_to_indices[label].append(idx)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")

        print(f"✅ Loaded {len(self.samples)} images from {self.name} dataset")
        print(f"   Number of labels: {len(self.label_to_indices)}")

    def get_samples_by_label(self, label: str) -> List[FaceSample]:
        """Get all samples for a specific label"""
        indices = self.label_to_indices.get(label, [])
        return [self.samples[i] for i in indices]

    def get_all_labels(self) -> List[str]:
        """Get all unique labels"""
        return list(self.label_to_indices.keys())

    def get_sample_count_per_label(self) -> Dict[str, int]:
        """Get count of samples per label"""
        return {label: len(indices) for label, indices in self.label_to_indices.items()}

    def split_dataset(self,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      seed: int = 42):
        """
        Split dataset into train, validation, and test sets
        Maintains class distribution
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        random.seed(seed)

        train_indices = []
        val_indices = []
        test_indices = []

        for label, indices in self.label_to_indices.items():
            # Shuffle indices for this label
            random.shuffle(indices)

            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        # Create dataset splits
        train_dataset = DatasetSplit(self, train_indices, "train")
        val_dataset = DatasetSplit(self, val_indices, "validation")
        test_dataset = DatasetSplit(self, test_indices, "test")

        print(f"📊 Dataset split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

        return train_dataset, val_dataset, test_dataset

    def create_cross_validation_folds(self, n_folds: int = 5, seed: int = 42):
        """
        Create cross-validation folds
        Returns list of (train_dataset, val_dataset) for each fold
        """
        random.seed(seed)
        folds = []

        for fold_idx in range(n_folds):
            train_indices = []
            val_indices = []

            for label, indices in self.label_to_indices.items():
                # Shuffle indices
                random.shuffle(indices)

                # Split for this fold
                fold_size = len(indices) // n_folds
                val_start = fold_idx * fold_size
                val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else len(indices)

                val_indices.extend(indices[val_start:val_end])
                train_indices.extend(indices[:val_start] + indices[val_end:])

            train_dataset = DatasetSplit(self, train_indices, f"train_fold_{fold_idx}")
            val_dataset = DatasetSplit(self, val_indices, f"val_fold_{fold_idx}")
            folds.append((train_dataset, val_dataset))

        print(f"📊 Created {n_folds} cross-validation folds")
        return folds

    def save_metadata(self, output_path: str):
        """Save dataset metadata to file"""
        metadata = {
            'name': self.name,
            'root_dir': str(self.root_dir),
            'num_samples': len(self.samples),
            'num_labels': len(self.label_to_indices),
            'samples_per_label': self.get_sample_count_per_label(),
            'samples': []
        }

        for sample in self.samples:
            sample_info = {
                'image_path': sample.image_path,
                'label': sample.label,
                'image_shape': sample.image.shape if sample.image is not None else None,
                'bbox': sample.bbox,
                'landmarks': sample.landmarks,
                'attributes': sample.attributes
            }
            metadata['samples'].append(sample_info)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Metadata saved to {output_path}")
        return metadata


class DatasetSplit:
    """
    Represents a split of the dataset (train/val/test)
    """

    def __init__(self, parent_dataset: FaceDataset, indices: List[int], split_name: str):
        self.parent_dataset = parent_dataset
        self.indices = indices
        self.split_name = split_name
        self.samples = [parent_dataset.samples[i] for i in indices]

        # Create label mapping for this split
        self.label_to_indices = {}
        for idx, sample_idx in enumerate(indices):
            label = parent_dataset.samples[sample_idx].label
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> FaceSample:
        return self.samples[idx]

    def get_batch(self, indices: List[int]) -> List[FaceSample]:
        """Get batch of samples"""
        return [self.samples[i] for i in indices]

    def get_stats(self) -> Dict:
        """Get statistics for this split"""
        stats = {
            'split_name': self.split_name,
            'num_samples': len(self),
            'num_labels': len(self.label_to_indices),
            'samples_per_label': {label: len(indices)
                                  for label, indices in self.label_to_indices.items()}
        }
        return stats


class Preprocessor:
    """
    Image preprocessing pipeline for face recognition
    """

    def __init__(self):
        self.pipeline = []

    def add_resize(self, size: Tuple[int, int] = (160, 160)):
        """Add resize operation"""

        def resize(image: np.ndarray) -> np.ndarray:
            return cv2.resize(image, size)

        self.pipeline.append(('resize', resize))
        return self

    def add_normalize(self, mean: List[float] = None, std: List[float] = None):
        """Add normalization"""
        if mean is None:
            mean = [0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5]

        def normalize(image: np.ndarray) -> np.ndarray:
            image = image.astype(np.float32) / 255.0
            image = (image - mean) / std
            return image

        self.pipeline.append(('normalize', normalize))
        return self

    def add_random_crop(self, crop_size: Tuple[int, int] = (144, 144)):
        """Add random crop augmentation"""

        def random_crop(image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            ch, cw = crop_size

            if h < ch or w < cw:
                return image

            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            return image[y:y + ch, x:x + cw]

        self.pipeline.append(('random_crop', random_crop))
        return self

    def add_random_flip(self, probability: float = 0.5):
        """Add random horizontal flip"""

        def random_flip(image: np.ndarray) -> np.ndarray:
            if random.random() < probability:
                return cv2.flip(image, 1)
            return image

        self.pipeline.append(('random_flip', random_flip))
        return self

    def add_color_jitter(self,
                         brightness: float = 0.2,
                         contrast: float = 0.2,
                         saturation: float = 0.2):
        """Add color jitter augmentation"""

        def color_jitter(image: np.ndarray) -> np.ndarray:
            # Convert to HSV for easier manipulation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

            # Brightness
            if brightness > 0:
                hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + random.uniform(-brightness, brightness))

            # Contrast (through value channel)
            if contrast > 0:
                hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + random.uniform(-contrast, contrast))

            # Saturation
            if saturation > 0:
                hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + random.uniform(-saturation, saturation))

            # Clip values
            hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)  # Hue range in OpenCV

            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        self.pipeline.append(('color_jitter', color_jitter))
        return self

    def process(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Process image through pipeline
        Args:
            image: Input image
            augment: Whether to apply augmentation operations
        """
        result = image.copy()

        for name, operation in self.pipeline:
            # Skip augmentation operations if not augmenting
            if not augment and name in ['random_crop', 'random_flip', 'color_jitter']:
                continue
            result = operation(result)

        return result

    def process_batch(self, images: List[np.ndarray], augment: bool = False) -> List[np.ndarray]:
        """Process batch of images"""
        return [self.process(img, augment) for img in images]


class DataGenerator:
    """
    Data generator for training face recognition models
    Supports batch generation with various sampling strategies
    """

    def __init__(self,
                 dataset: Union[FaceDataset, DatasetSplit],
                 batch_size: int = 32,
                 preprocessor: Preprocessor = None,
                 shuffle: bool = True,
                 seed: int = 42):

        self.dataset = dataset
        self.batch_size = batch_size
        self.preprocessor = preprocessor or Preprocessor()
        self.shuffle = shuffle
        self.seed = seed

        self.indices = list(range(len(dataset)))
        self.current_index = 0

        if shuffle:
            random.seed(seed)
            random.shuffle(self.indices)

    def __len__(self) -> int:
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[List[np.ndarray], List[str]]:
        """Get next batch"""
        if self.current_index >= len(self.indices):
            raise StopIteration

        # Get batch indices
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        # Get samples
        batch_samples = [self.dataset[i] for i in batch_indices]

        # Extract images and labels
        images = [sample.image for sample in batch_samples]
        labels = [sample.label for sample in batch_samples]

        # Preprocess images
        processed_images = self.preprocessor.process_batch(images, augment=self.shuffle)

        return processed_images, labels

    def create_triplet_batch(self, n_triplets: int = 16):
        """
        Create triplet batch for metric learning
        Each triplet: (anchor, positive, negative)
        """
        triplets = []

        # Get all labels
        labels = list(self.dataset.label_to_indices.keys())

        for _ in range(n_triplets):
            # Random anchor label
            anchor_label = random.choice(labels)
            anchor_indices = self.dataset.label_to_indices[anchor_label]

            # Choose anchor and positive from same label
            anchor_idx, positive_idx = random.sample(anchor_indices, 2)
            anchor_sample = self.dataset[anchor_idx]
            positive_sample = self.dataset[positive_idx]

            # Choose negative from different label
            negative_label = random.choice([l for l in labels if l != anchor_label])
            negative_indices = self.dataset.label_to_indices[negative_label]
            negative_idx = random.choice(negative_indices)
            negative_sample = self.dataset[negative_idx]

            # Preprocess
            anchor_img = self.preprocessor.process(anchor_sample.image, augment=True)
            positive_img = self.preprocessor.process(positive_sample.image, augment=True)
            negative_img = self.preprocessor.process(negative_sample.image, augment=True)

            triplets.append((anchor_img, positive_img, negative_img))

        return triplets

    def create_pair_batch(self, n_pairs: int = 32, positive_ratio: float = 0.5):
        """
        Create pairs for verification training
        Returns pairs and labels (1 for positive, 0 for negative)
        """
        pairs = []
        pair_labels = []

        labels = list(self.dataset.label_to_indices.keys())

        for _ in range(n_pairs):
            if random.random() < positive_ratio:
                # Positive pair
                label = random.choice(labels)
                indices = self.dataset.label_to_indices[label]
                idx1, idx2 = random.sample(indices, 2)

                sample1 = self.dataset[idx1]
                sample2 = self.dataset[idx2]
                pair_labels.append(1)
            else:
                # Negative pair
                label1, label2 = random.sample(labels, 2)
                idx1 = random.choice(self.dataset.label_to_indices[label1])
                idx2 = random.choice(self.dataset.label_to_indices[label2])

                sample1 = self.dataset[idx1]
                sample2 = self.dataset[idx2]
                pair_labels.append(0)

            # Preprocess
            img1 = self.preprocessor.process(sample1.image, augment=True)
            img2 = self.preprocessor.process(sample2.image, augment=True)

            pairs.append((img1, img2))

        return pairs, pair_labels


# Example usage and testing
if __name__ == "__main__":
    # Test data loader
    print("🧪 Testing Data Loader...")

    # Create dummy dataset
    dummy_images = []
    for i in range(10):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        dummy_images.append(img)

    # Create a simple test
    print("✅ Data loader module loaded successfully")
    print("\nAvailable classes:")
    print("1. FaceDataset - Load and manage face datasets")
    print("2. Preprocessor - Image preprocessing pipeline")
    print("3. DataGenerator - Batch generator for training")
    print("4. FaceSample - Data class for face samples")

    # Test preprocessor
    preprocessor = Preprocessor()
    preprocessor.add_resize((160, 160))
    preprocessor.add_normalize()

    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    processed = preprocessor.process(test_image)

    print(f"\n📊 Preprocessor test:")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {processed.shape}")
    print(f"  Output dtype: {processed.dtype}")

    print("\n✅ Data utilities ready for use!")

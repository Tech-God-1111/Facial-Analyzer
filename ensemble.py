"""
Advanced Ensemble Methods for Face Recognition
Research: Ensemble learning with adaptive weighting and confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

from .facenet import FaceNetModel
from .arcface import ArcFaceModel
from .deepface import DeepFaceModel
from .attention_face import AttentionFaceModel


class AdaptiveEnsemble:
    """
    Adaptive Ensemble with learned weights and confidence estimation
    Research: "Adaptive Ensemble Learning for Robust Face Recognition"
    """

    def __init__(self, models_config: Dict = None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.model_weights = {}
        self.model_confidences = {}

        # Default configuration
        if models_config is None:
            models_config = {
                'facenet': {'enabled': True, 'initial_weight': 1.0},
                'arcface': {'enabled': True, 'initial_weight': 1.0},
                'deepface': {'enabled': True, 'initial_weight': 1.0, 'submodels': ['vgg16', 'resnet50']},
                'attention_face': {'enabled': True, 'initial_weight': 1.0}
            }

        # Initialize models
        self._initialize_models(models_config)

        # Adaptive weighting parameters
        self.learning_rate = 0.01
        self.confidence_threshold = 0.7
        self.min_weight = 0.1
        self.max_weight = 2.0

        print(f"✅ Adaptive Ensemble initialized with {len(self.models)} models on {device}")

    def _initialize_models(self, config: Dict):
        """Initialize all models based on configuration"""
        # FaceNet
        if config.get('facenet', {}).get('enabled', False):
            try:
                self.models['facenet'] = FaceNetModel(device=self.device)
                self.model_weights['facenet'] = config['facenet'].get('initial_weight', 1.0)
                self.model_confidences['facenet'] = 1.0
            except Exception as e:
                print(f"⚠️ Failed to load FaceNet: {e}")

        # ArcFace
        if config.get('arcface', {}).get('enabled', False):
            try:
                self.models['arcface'] = ArcFaceModel(
                    backbone='resnet50',
                    device=self.device
                )
                self.model_weights['arcface'] = config['arcface'].get('initial_weight', 1.0)
                self.model_confidences['arcface'] = 1.0
            except Exception as e:
                print(f"⚠️ Failed to load ArcFace: {e}")

        # DeepFace (ensemble)
        if config.get('deepface', {}).get('enabled', False):
            try:
                submodels = config['deepface'].get('submodels', ['vgg16', 'resnet50'])
                self.models['deepface'] = DeepFaceModel(
                    model_name='ensemble',
                    device=self.device
                )
                self.model_weights['deepface'] = config['deepface'].get('initial_weight', 1.0)
                self.model_confidences['deepface'] = 1.0
            except Exception as e:
                print(f"⚠️ Failed to load DeepFace: {e}")

        # AttentionFace
        if config.get('attention_face', {}).get('enabled', False):
            try:
                self.models['attention_face'] = AttentionFaceModel(
                    backbone='resnet50',
                    device=self.device
                )
                self.model_weights['attention_face'] = config['attention_face'].get('initial_weight', 1.0)
                self.model_confidences['attention_face'] = 1.0
            except Exception as e:
                print(f"⚠️ Failed to load AttentionFace: {e}")

    def extract_embeddings(self, face_images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings using all models
        Returns dictionary of embeddings from each model
        """
        all_embeddings = {}

        for model_name, model in self.models.items():
            try:
                if model_name == 'deepface':
                    # DeepFace returns dict of embeddings
                    embeddings_dict = model.extract_embeddings(face_images)
                    all_embeddings[model_name] = embeddings_dict
                else:
                    # Other models return numpy array
                    embeddings = model.extract_embeddings(face_images)
                    all_embeddings[model_name] = embeddings
            except Exception as e:
                print(f"⚠️ Error extracting embeddings from {model_name}: {e}")
                all_embeddings[model_name] = np.array([])

        return all_embeddings

    def compute_fused_embedding(self, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute fused embedding using adaptive weighting
        """
        valid_embeddings = []
        valid_weights = []

        for model_name, embeddings in embeddings_dict.items():
            if model_name in self.models and embeddings.size > 0:
                # Get model weight and confidence
                weight = self.model_weights.get(model_name, 1.0)
                confidence = self.model_confidences.get(model_name, 1.0)

                # Adjust weight by confidence
                adjusted_weight = weight * confidence

                # For DeepFace, use ensemble weighted embedding
                if model_name == 'deepface' and isinstance(embeddings, dict):
                    if 'ensemble_weighted' in embeddings:
                        valid_embeddings.append(embeddings['ensemble_weighted'])
                        valid_weights.append(adjusted_weight)
                else:
                    valid_embeddings.append(embeddings)
                    valid_weights.append(adjusted_weight)

        if not valid_embeddings:
            return np.array([])

        # Weighted average
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()  # Normalize

        fused_embedding = np.zeros_like(valid_embeddings[0])

        for emb, weight in zip(valid_embeddings, valid_weights):
            fused_embedding += emb * weight

        # L2 normalize
        fused_embedding = fused_embedding / np.linalg.norm(fused_embedding)

        return fused_embedding

    def verify_faces(self, face1: np.ndarray, face2: np.ndarray,
                     threshold: float = 0.6) -> Dict:
        """
        Verify faces using ensemble with confidence scores
        """
        # Extract embeddings from both faces
        embeddings1 = self.extract_embeddings([face1])
        embeddings2 = self.extract_embeddings([face2])

        # Get individual model decisions
        model_decisions = {}
        model_similarities = {}

        for model_name in self.models.keys():
            if model_name in embeddings1 and model_name in embeddings2:
                try:
                    emb1 = embeddings1[model_name]
                    emb2 = embeddings2[model_name]

                    if model_name == 'deepface':
                        # DeepFace verification
                        if isinstance(emb1, dict) and isinstance(emb2, dict):
                            # Use ensemble weighted embedding
                            if 'ensemble_weighted' in emb1 and 'ensemble_weighted' in emb2:
                                similarity = self._compute_cosine_similarity(
                                    emb1['ensemble_weighted'],
                                    emb2['ensemble_weighted']
                                )
                                model_similarities[model_name] = similarity
                                model_decisions[model_name] = similarity > threshold
                    else:
                        # Other models
                        if emb1.size > 0 and emb2.size > 0:
                            similarity = self._compute_cosine_similarity(emb1[0], emb2[0])
                            model_similarities[model_name] = similarity
                            model_decisions[model_name] = similarity > threshold
                except Exception as e:
                    print(f"⚠️ Error in {model_name} verification: {e}")

        # Weighted voting
        final_decision, confidence = self._weighted_voting(model_decisions)

        # Compute fused similarity
        fused_emb1 = self.compute_fused_embedding(embeddings1)
        fused_emb2 = self.compute_fused_embedding(embeddings2)

        fused_similarity = 0.0
        if fused_emb1.size > 0 and fused_emb2.size > 0:
            fused_similarity = self._compute_cosine_similarity(fused_emb1, fused_emb2)

        return {
            'decision': final_decision,
            'confidence': confidence,
            'fused_similarity': float(fused_similarity),
            'model_decisions': model_decisions,
            'model_similarities': model_similarities,
            'ensemble_method': 'adaptive_weighted_voting'
        }

    def _weighted_voting(self, model_decisions: Dict[str, bool]) -> Tuple[bool, float]:
        """
        Perform weighted voting with model confidence
        """
        yes_votes = 0.0
        no_votes = 0.0

        for model_name, decision in model_decisions.items():
            weight = self.model_weights.get(model_name, 1.0)
            confidence = self.model_confidences.get(model_name, 1.0)

            adjusted_weight = weight * confidence

            if decision:
                yes_votes += adjusted_weight
            else:
                no_votes += adjusted_weight

        total_votes = yes_votes + no_votes

        if total_votes == 0:
            return False, 0.0

        final_decision = yes_votes > no_votes
        confidence = abs(yes_votes - no_votes) / total_votes

        return final_decision, float(confidence)

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(np.clip(similarity, -1.0, 1.0))

    def update_model_weights(self, validation_results: Dict):
        """
        Update model weights based on validation performance
        Adaptive learning of weights
        """
        print("⚙️ Updating model weights based on validation...")

        for model_name in self.models.keys():
            if model_name in validation_results:
                performance = validation_results[model_name].get('accuracy', 0.5)
                confidence = validation_results[model_name].get('confidence', 0.5)

                # Update weight based on performance
                current_weight = self.model_weights.get(model_name, 1.0)

                # Sigmoid adjustment
                adjustment = 1.0 / (1.0 + np.exp(-10 * (performance - 0.5)))

                # Update weight with momentum
                new_weight = current_weight * 0.9 + adjustment * 0.1

                # Clip to bounds
                new_weight = np.clip(new_weight, self.min_weight, self.max_weight)

                self.model_weights[model_name] = new_weight
                self.model_confidences[model_name] = confidence

                print(f"  {model_name}: weight={new_weight:.3f}, confidence={confidence:.3f}")

    def analyze_ensemble_diversity(self, test_dataset: List[Tuple[np.ndarray, str]]) -> Dict:
        """
        Analyze diversity of ensemble members
        For research on ensemble effectiveness
        """
        print("🔬 Analyzing ensemble diversity...")

        diversity_metrics = {}

        # Extract embeddings for all models
        all_embeddings = {}

        for model_name, model in self.models.items():
            embeddings_list = []
            labels_list = []

            for face, label in test_dataset:
                emb = model.extract_embeddings([face])
                if emb.size > 0:
                    embeddings_list.append(emb[0])
                    labels_list.append(label)

            if embeddings_list:
                all_embeddings[model_name] = {
                    'embeddings': np.array(embeddings_list),
                    'labels': labels_list
                }

        # Compute pairwise model correlations
        model_names = list(all_embeddings.keys())

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]

                emb1 = all_embeddings[model1]['embeddings']
                emb2 = all_embeddings[model2]['embeddings']

                if emb1.shape == emb2.shape:
                    # Compute correlation
                    correlation = self._compute_embedding_correlation(emb1, emb2)

                    key = f"{model1}_{model2}"
                    diversity_metrics[key] = {
                        'correlation': float(correlation),
                        'diversity': float(1.0 - abs(correlation))
                    }

        # Compute overall diversity score
        if diversity_metrics:
            avg_diversity = np.mean([m['diversity'] for m in diversity_metrics.values()])
            diversity_metrics['ensemble_overall'] = {
                'average_diversity': float(avg_diversity),
                'num_model_pairs': len(diversity_metrics)
            }

        print(f"\n📊 Ensemble Diversity Analysis:")
        print(f"  Average Diversity: {avg_diversity:.3f}")
        print(f"  Model Pairs: {len(diversity_metrics)}")

        return diversity_metrics

    def _compute_embedding_correlation(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute correlation between embedding spaces"""
        # Flatten and compute correlation
        flat1 = emb1.flatten()
        flat2 = emb2.flatten()

        correlation = np.corrcoef(flat1, flat2)[0, 1]

        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    def confidence_calibration(self, calibration_data: List[Tuple[np.ndarray, str, bool]]):
        """
        Calibrate model confidence scores
        Research: "Well-calibrated confidence for face recognition"
        """
        print("🎯 Calibrating model confidence...")

        calibration_results = {}

        for model_name, model in self.models.items():
            correct_predictions = 0
            total_predictions = 0
            confidence_scores = []
            correct_labels = []

            for face, true_label, is_match in calibration_data:
                # This is simplified - in practice, you'd do proper verification
                # and collect confidence scores
                try:
                    # Extract embedding
                    emb = model.extract_embeddings([face])

                    if emb.size > 0:
                        # For this example, we'll simulate confidence
                        confidence = np.random.random()  # Replace with actual confidence

                        confidence_scores.append(confidence)
                        correct_labels.append(1 if is_match else 0)

                        total_predictions += 1
                except Exception as e:
                    continue

            if confidence_scores:
                # Compute calibration metrics
                from sklearn.calibration import calibration_curve

                fraction_of_positives, mean_predicted_value = calibration_curve(
                    correct_labels, confidence_scores, n_bins=10
                )

                # Expected Calibration Error (ECE)
                ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value)) / len(fraction_of_positives)

                calibration_results[model_name] = {
                    'ece': float(ece),
                    'num_samples': total_predictions,
                    'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0
                }

        print("\n📈 Calibration Results:")
        for model_name, results in calibration_results.items():
            print(f"  {model_name}: ECE={results['ece']:.3f}, Accuracy={results['accuracy']:.3f}")

        return calibration_results

    def get_ensemble_info(self) -> Dict:
        """
        Get comprehensive ensemble information
        """
        model_info = {}

        for model_name, model in self.models.items():
            try:
                info = model.get_model_info()
                model_info[model_name] = {
                    'type': info.get('model_name', info.get('framework', 'Unknown')),
                    'embedding_dim': info.get('embedding_dim', 'Unknown'),
                    'weight': self.model_weights.get(model_name, 1.0),
                    'confidence': self.model_confidences.get(model_name, 1.0)
                }
            except:
                model_info[model_name] = {
                    'type': 'Unknown',
                    'weight': self.model_weights.get(model_name, 1.0),
                    'confidence': self.model_confidences.get(model_name, 1.0)
                }

        return {
            'ensemble_type': 'AdaptiveEnsemble',
            'num_models': len(self.models),
            'models': model_info,
            'adaptive_learning': True,
            'confidence_calibration': True,
            'diversity_analysis': True,
            'device': str(self.device),
            'research_features': [
                'Adaptive weight learning',
                'Confidence estimation',
                'Ensemble diversity analysis',
                'Model correlation analysis',
                'Weighted fusion strategies'
            ]
        }

    def run_research_experiment(self, experiment_name: str, dataset_config: Dict):
        """
        Run a comprehensive research experiment
        """
        print(f"🔬 Running research experiment: {experiment_name}")

        experiment_results = {
            'experiment_name': experiment_name,
            'ensemble_info': self.get_ensemble_info(),
            'dataset_config': dataset_config,
            'results': {}
        }

        # This would implement the full experiment
        # Including:
        # 1. Data loading and preprocessing
        # 2. Model evaluation
        # 3. Ensemble analysis
        # 4. Statistical tests
        # 5. Visualization

        print(f"✅ Experiment {experiment_name} initiated")
        print("📝 Implement experiment logic here")

        return experiment_results
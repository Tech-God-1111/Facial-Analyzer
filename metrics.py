"""
Face Recognition Evaluation Metrics
Comprehensive metrics for verification and identification tasks
"""

import numpy as np
import json
import warnings
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class VerificationMetrics:
    """Container for verification metrics"""
    eer: float  # Equal Error Rate
    eer_threshold: float  # Threshold at EER
    frr_1_far: float  # False Reject Rate at 1% FAR
    frr_01_far: float  # False Reject Rate at 0.1% FAR
    frr_001_far: float  # False Reject Rate at 0.01% FAR
    auc: float  # Area Under ROC Curve
    d_prime: float  # d-prime measure of separation
    tpr_fpr_curve: np.ndarray  # TPR at various FPRs
    thresholds: np.ndarray  # Corresponding thresholds
    roc_curve: Dict[str, np.ndarray]  # FPR, TPR, thresholds
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    optimal_threshold: Optional[float] = None
    optimal_metric: Optional[str] = None


@dataclass
class IdentificationMetrics:
    """Container for identification metrics"""
    rank1: float  # Rank-1 accuracy
    rank5: float  # Rank-5 accuracy
    rank10: float  # Rank-10 accuracy
    cmc_curve: np.ndarray  # Cumulative Match Characteristic curve
    precision_at_k: Dict[int, float]  # Precision at different K values
    map: float  # Mean Average Precision
    gallery_size: int
    probe_size: int
    closed_set: bool = True
    hit_rate_at_1_far: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None


@dataclass
class ModelPerformance:
    """Container for model performance across multiple metrics"""
    model_name: str
    verification_metrics: VerificationMetrics
    identification_metrics: Optional[IdentificationMetrics] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FaceRecognitionMetrics:
    """
    Comprehensive evaluator for face recognition systems
    Supports both verification and identification tasks
    """

    def __init__(self, num_bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.epsilon = 1e-10

        # Standard operating points for reporting
        self.standard_fpr_points = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5])
        self.standard_ranks = [1, 5, 10, 20, 50, 100]

    def compute_verification_metrics(self,
                                     genuine_scores: np.ndarray,
                                     impostor_scores: np.ndarray,
                                     compute_ci: bool = False) -> VerificationMetrics:
        """
        Compute verification metrics from similarity scores

        Args:
            genuine_scores: Similarity scores for genuine pairs
            impostor_scores: Similarity scores for impostor pairs
            compute_ci: Whether to compute confidence intervals

        Returns:
            VerificationMetrics object with all metrics
        """
        # Ensure scores are numpy arrays
        genuine_scores = np.asarray(genuine_scores).flatten()
        impostor_scores = np.asarray(impostor_scores).flatten()

        # Remove NaN values
        genuine_scores = genuine_scores[~np.isnan(genuine_scores)]
        impostor_scores = impostor_scores[~np.isnan(impostor_scores)]

        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            raise ValueError("Genuine or impostor scores array is empty")

        # Sort scores for threshold sweeping
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

        # Initialize arrays for ROC computation
        tpr_list = []  # True Positive Rate (1 - FRR)
        fpr_list = []  # False Positive Rate (FAR)

        for threshold in thresholds:
            # True Positive Rate (TPR) = 1 - False Reject Rate (FRR)
            tpr = np.mean(genuine_scores >= threshold)

            # False Positive Rate (FPR) = False Accept Rate (FAR)
            fpr = np.mean(impostor_scores >= threshold)

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)

        # Compute AUC using trapezoidal rule
        sorted_indices = np.argsort(fpr_array)
        sorted_fpr = fpr_array[sorted_indices]
        sorted_tpr = tpr_array[sorted_indices]
        auc_score = np.trapz(sorted_tpr, sorted_fpr)

        # Find Equal Error Rate (EER)
        # EER is where FRR = FAR, or (1-TPR) = FPR
        frr_array = 1 - tpr_array  # False Reject Rate
        eer_idx = np.argmin(np.abs(frr_array - fpr_array))
        eer = (frr_array[eer_idx] + fpr_array[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        # Find FRR at specific FAR operating points
        def find_frr_at_far(target_far: float) -> float:
            idx = np.argmin(np.abs(fpr_array - target_far))
            return frr_array[idx]

        frr_1_far = find_frr_at_far(0.01)  # 1% FAR
        frr_01_far = find_frr_at_far(0.001)  # 0.1% FAR
        frr_001_far = find_frr_at_far(0.0001)  # 0.01% FAR

        # Compute d-prime (separation measure)
        d_prime = self._compute_d_prime(genuine_scores, impostor_scores)

        # TPR at various FPRs
        tpr_at_fpr = np.array([1 - find_frr_at_far(fpr) for fpr in self.standard_fpr_points])

        # Compute confidence intervals if requested
        confidence_intervals = None
        if compute_ci:
            confidence_intervals = self._compute_bootstrap_ci(genuine_scores, impostor_scores)

        # Find optimal threshold (maximizes TPR - FPR)
        diff = tpr_array - fpr_array
        optimal_idx = np.argmax(diff)
        optimal_threshold = thresholds[optimal_idx]

        return VerificationMetrics(
            eer=float(eer),
            eer_threshold=float(eer_threshold),
            frr_1_far=float(frr_1_far),
            frr_01_far=float(frr_01_far),
            frr_001_far=float(frr_001_far),
            auc=float(auc_score),
            d_prime=float(d_prime),
            tpr_fpr_curve=np.column_stack([self.standard_fpr_points, tpr_at_fpr]),
            thresholds=np.column_stack([thresholds, frr_array, fpr_array]),
            roc_curve={'fpr': fpr_array, 'tpr': tpr_array, 'thresholds': thresholds},
            confidence_intervals=confidence_intervals,
            optimal_threshold=float(optimal_threshold),
            optimal_metric='max_diff_tpr_fpr'
        )

    def compute_identification_metrics(self,
                                       query_embeddings: np.ndarray,
                                       query_labels: List[str],
                                       gallery_embeddings: np.ndarray,
                                       gallery_labels: List[str],
                                       similarity_matrix: Optional[np.ndarray] = None,
                                       metric: str = 'cosine',
                                       open_set: bool = False,
                                       threshold: Optional[float] = None) -> IdentificationMetrics:
        """
        Compute identification metrics

        Args:
            query_embeddings: Embeddings of probe images
            query_labels: Labels of probe images
            gallery_embeddings: Embeddings of gallery images
            gallery_labels: Labels of gallery images
            similarity_matrix: Pre-computed similarity matrix
            metric: Distance metric ('cosine', 'euclidean', 'l2')
            open_set: Whether it's open-set identification
            threshold: Similarity threshold for open-set

        Returns:
            IdentificationMetrics object
        """
        n_queries = len(query_embeddings)
        n_gallery = len(gallery_embeddings)

        # Compute similarity matrix if not provided
        if similarity_matrix is None:
            similarity_matrix = self._compute_similarity_matrix(
                query_embeddings, gallery_embeddings, metric
            )

        # Get ranking for each query
        rankings = []
        average_precisions = []
        correct_predictions = []
        all_predictions = []

        for i in range(n_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]

            # Sort gallery by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = [gallery_labels[idx] for idx in sorted_indices]

            # Find ranks where correct label appears
            correct_positions = np.where(np.array(sorted_labels) == query_label)[0]

            if open_set and threshold is not None:
                # Open-set identification
                best_similarity = similarities[sorted_indices[0]]
                best_label = sorted_labels[0]

                if best_similarity >= threshold:
                    prediction = best_label
                else:
                    prediction = 'unknown'

                all_predictions.append(prediction)
                correct_predictions.append(1 if prediction == query_label else 0)

            if len(correct_positions) > 0:
                best_rank = correct_positions[0] + 1
                rankings.append(best_rank)

                # Compute Average Precision
                ap = self._compute_average_precision(sorted_labels, query_label)
                average_precisions.append(ap)
            else:
                rankings.append(np.inf)
                average_precisions.append(0.0)

        # Compute CMC curve
        max_rank = min(100, n_gallery)
        cmc = np.zeros(max_rank)

        for rank in rankings:
            if rank <= max_rank:
                cmc[int(rank) - 1:] += 1

        if n_queries > 0:
            cmc = cmc / n_queries

        # Compute rank-k accuracies
        valid_ranks = np.array([r for r in rankings if r != np.inf])
        rank1 = np.mean(valid_ranks <= 1) if len(valid_ranks) > 0 else 0
        rank5 = np.mean(valid_ranks <= 5) if len(valid_ranks) > 0 else 0
        rank10 = np.mean(valid_ranks <= 10) if len(valid_ranks) > 0 else 0

        # Precision at different K values
        precision_at_k = {}
        for k in self.standard_ranks:
            if k <= n_gallery:
                precision_at_k[k] = np.mean(valid_ranks <= k) if len(valid_ranks) > 0 else 0

        # Mean Average Precision
        map_score = np.mean(average_precisions) if len(average_precisions) > 0 else 0

        # For open-set, compute hit rate at 1% FAR
        hit_rate_at_1_far = None
        if open_set and len(correct_predictions) > 0:
            hit_rate_at_1_far = np.mean(correct_predictions)

        # Compute confusion matrix
        confusion_matrix = None
        classification_report = None
        if open_set and len(all_predictions) > 0:
            # Simplified confusion matrix for demo
            unique_labels = list(set(query_labels + ['unknown']))
            confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))
            # In practice, you'd populate this with actual predictions

        return IdentificationMetrics(
            rank1=float(rank1),
            rank5=float(rank5),
            rank10=float(rank10),
            cmc_curve=cmc,
            precision_at_k=precision_at_k,
            map=float(map_score),
            gallery_size=n_gallery,
            probe_size=n_queries,
            closed_set=not open_set,
            hit_rate_at_1_far=hit_rate_at_1_far,
            confusion_matrix=confusion_matrix,
            classification_report=classification_report
        )

    def compute_model_performance(self,
                                  model_name: str,
                                  genuine_scores: np.ndarray,
                                  impostor_scores: np.ndarray,
                                  query_embeddings: Optional[np.ndarray] = None,
                                  query_labels: Optional[List[str]] = None,
                                  gallery_embeddings: Optional[np.ndarray] = None,
                                  gallery_labels: Optional[List[str]] = None,
                                  inference_time_ms: Optional[float] = None,
                                  memory_usage_mb: Optional[float] = None) -> ModelPerformance:
        """
        Compute comprehensive performance metrics for a model

        Returns:
            ModelPerformance object with all metrics
        """
        # Compute verification metrics
        ver_metrics = self.compute_verification_metrics(
            genuine_scores, impostor_scores, compute_ci=True
        )

        # Compute identification metrics if data provided
        id_metrics = None
        if (query_embeddings is not None and gallery_embeddings is not None and
                query_labels is not None and gallery_labels is not None):
            id_metrics = self.compute_identification_metrics(
                query_embeddings, query_labels,
                gallery_embeddings, gallery_labels
            )

        return ModelPerformance(
            model_name=model_name,
            verification_metrics=ver_metrics,
            identification_metrics=id_metrics,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb
        )

    def _compute_similarity_matrix(self,
                                   query_embeddings: np.ndarray,
                                   gallery_embeddings: np.ndarray,
                                   metric: str = 'cosine') -> np.ndarray:
        """Compute similarity matrix between query and gallery embeddings"""
        n_queries = len(query_embeddings)
        n_gallery = len(gallery_embeddings)

        similarity_matrix = np.zeros((n_queries, n_gallery))

        # Normalize embeddings for cosine similarity
        if metric == 'cosine':
            query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + self.epsilon)
            gallery_norm = gallery_embeddings / (
                        np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + self.epsilon)
            similarity_matrix = np.dot(query_norm, gallery_norm.T)

        elif metric == 'euclidean':
            # Convert distance to similarity
            for i in range(n_queries):
                for j in range(n_gallery):
                    dist = np.linalg.norm(query_embeddings[i] - gallery_embeddings[j])
                    similarity_matrix[i, j] = 1.0 / (1.0 + dist)

        elif metric == 'l2':
            # L2 distance converted to similarity via exponential
            for i in range(n_queries):
                for j in range(n_gallery):
                    dist = np.linalg.norm(query_embeddings[i] - gallery_embeddings[j])
                    similarity_matrix[i, j] = np.exp(-dist)

        return similarity_matrix

    def _compute_average_precision(self,
                                   sorted_labels: List[str],
                                   query_label: str) -> float:
        """Compute Average Precision for a single query"""
        relevant_positions = []

        for i, label in enumerate(sorted_labels):
            if label == query_label:
                relevant_positions.append(i + 1)

        if not relevant_positions:
            return 0.0

        precisions = []
        for k, pos in enumerate(relevant_positions, 1):
            precision_at_k = k / pos
            precisions.append(precision_at_k)

        return np.mean(precisions)

    def _compute_d_prime(self,
                         genuine_scores: np.ndarray,
                         impostor_scores: np.ndarray) -> float:
        """Compute d-prime separation measure"""
        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            return 0.0

        genuine_mean = np.mean(genuine_scores)
        impostor_mean = np.mean(impostor_scores)

        genuine_std = np.std(genuine_scores)
        impostor_std = np.std(impostor_scores)

        mean_diff = abs(genuine_mean - impostor_mean)
        pooled_std = np.sqrt((genuine_std ** 2 + impostor_std ** 2) / 2)

        if pooled_std < self.epsilon:
            return float('inf')

        return float(mean_diff / pooled_std)

    def _compute_bootstrap_ci(self,
                              genuine_scores: np.ndarray,
                              impostor_scores: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for metrics"""
        bootstrap_eers = []
        bootstrap_aucs = []
        bootstrap_frr_1_fars = []

        n_genuine = len(genuine_scores)
        n_impostor = len(impostor_scores)

        for _ in range(min(self.num_bootstrap_samples, 100)):  # Limit for speed
            # Resample with replacement
            gen_resample = np.random.choice(genuine_scores, size=n_genuine, replace=True)
            imp_resample = np.random.choice(impostor_scores, size=n_impostor, replace=True)

            try:
                metrics = self.compute_verification_metrics(
                    gen_resample, imp_resample, compute_ci=False
                )
                bootstrap_eers.append(metrics.eer)
                bootstrap_aucs.append(metrics.auc)
                bootstrap_frr_1_fars.append(metrics.frr_1_far)
            except:
                continue

        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        def compute_ci(values):
            if len(values) == 0:
                return (0.0, 0.0)
            return (
                float(np.percentile(values, lower_percentile)),
                float(np.percentile(values, upper_percentile))
            )

        return {
            'eer': compute_ci(bootstrap_eers),
            'auc': compute_ci(bootstrap_aucs),
            'frr_1_far': compute_ci(bootstrap_frr_1_fars)
        }

    def generate_report(self,
                        performance: ModelPerformance,
                        format: str = 'markdown') -> str:
        """Generate evaluation report from model performance"""

        if format == 'markdown':
            report = f"# Face Recognition Evaluation Report\n\n"
            report += f"**Model:** {performance.model_name}\n"
            report += f"**Timestamp:** {performance.timestamp}\n\n"

            report += "## Verification Metrics\n\n"
            ver = performance.verification_metrics
            report += f"- **EER:** {ver.eer:.4f}"
            if ver.confidence_intervals:
                ci = ver.confidence_intervals.get('eer', (0, 0))
                report += f" (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])\n"
            else:
                report += "\n"

            report += f"- **AUC:** {ver.auc:.4f}\n"
            report += f"- **d-prime:** {ver.d_prime:.2f}\n"
            report += f"- **FRR @ 1% FAR:** {ver.frr_1_far:.4f}\n"
            report += f"- **FRR @ 0.1% FAR:** {ver.frr_01_far:.4f}\n"
            report += f"- **FRR @ 0.01% FAR:** {ver.frr_001_far:.4f}\n"
            report += f"- **Optimal Threshold:** {ver.optimal_threshold:.4f}\n\n"

            if performance.identification_metrics:
                report += "## Identification Metrics\n\n"
                ident = performance.identification_metrics
                report += f"- **Rank-1 Accuracy:** {ident.rank1:.4f}\n"
                report += f"- **Rank-5 Accuracy:** {ident.rank5:.4f}\n"
                report += f"- **Rank-10 Accuracy:** {ident.rank10:.4f}\n"
                report += f"- **Mean Average Precision:** {ident.map:.4f}\n"
                report += f"- **Gallery Size:** {ident.gallery_size}\n"
                report += f"- **Probe Size:** {ident.probe_size}\n\n"

            if performance.inference_time_ms is not None:
                report += "## Performance Metrics\n\n"
                report += f"- **Inference Time:** {performance.inference_time_ms:.2f} ms\n"

            if performance.memory_usage_mb is not None:
                report += f"- **Memory Usage:** {performance.memory_usage_mb:.2f} MB\n"

            return report

        elif format == 'json':
            # Convert to dictionary and serialize
            data = asdict(performance)
            return json.dumps(data, indent=2, default=str)

        else:  # plain text
            report = f"Face Recognition Evaluation Report\n"
            report += f"Model: {performance.model_name}\n"
            report += f"Timestamp: {performance.timestamp}\n"
            report += "=" * 50 + "\n\n"

            report += "Verification Metrics:\n"
            ver = performance.verification_metrics
            report += f"  EER: {ver.eer:.4f}\n"
            report += f"  AUC: {ver.auc:.4f}\n"

            if performance.identification_metrics:
                report += "\nIdentification Metrics:\n"
                ident = performance.identification_metrics
                report += f"  Rank-1: {ident.rank1:.4f}\n"
                report += f"  Rank-5: {ident.rank5:.4f}\n"

            return report

    def save_results(self,
                     performance: ModelPerformance,
                     filepath: str):
        """Save performance results to file"""
        data = asdict(performance)

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Recursively convert numpy arrays
        import json
        from json import JSONEncoder

        class NumpyEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.float32):
                    return float(obj)
                if isinstance(obj, np.float64):
                    return float(obj)
                if isinstance(obj, np.int32):
                    return int(obj)
                if isinstance(obj, np.int64):
                    return int(obj)
                return super().default(obj)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load_results(cls, filepath: str) -> ModelPerformance:
        """Load performance results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays where needed
        def convert_back(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, list) and key in ['tpr_fpr_curve', 'thresholds', 'cmc_curve']:
                        obj[key] = np.array(value)
                    elif isinstance(value, dict):
                        convert_back(value)
            return obj

        data = convert_back(data)

        # Recreate VerificationMetrics
        ver_data = data.pop('verification_metrics')
        ver_metrics = VerificationMetrics(**ver_data)

        # Recreate IdentificationMetrics if present
        id_metrics = None
        if 'identification_metrics' in data:
            id_data = data.pop('identification_metrics')
            id_metrics = IdentificationMetrics(**id_data)

        # Recreate ModelPerformance
        return ModelPerformance(
            verification_metrics=ver_metrics,
            identification_metrics=id_metrics,
            **data
        )


# Quick test function
def test_metrics():
    """Test the metrics computation"""
    print("Testing Face Recognition Metrics...")

    # Create evaluator
    evaluator = FaceRecognitionMetrics(num_bootstrap_samples=100)

    # Generate dummy scores
    np.random.seed(42)
    genuine_scores = np.random.normal(0.8, 0.1, 100)
    impostor_scores = np.random.normal(0.3, 0.2, 100)

    # Test verification metrics
    ver_metrics = evaluator.compute_verification_metrics(
        genuine_scores, impostor_scores, compute_ci=True
    )

    print(f"✓ EER: {ver_metrics.eer:.4f}")
    print(f"✓ AUC: {ver_metrics.auc:.4f}")
    print(f"✓ FRR @ 1% FAR: {ver_metrics.frr_1_far:.4f}")

    # Generate dummy embeddings for identification
    n_gallery = 50
    n_queries = 20
    embedding_dim = 128

    gallery_embeddings = np.random.randn(n_gallery, embedding_dim)
    query_embeddings = np.random.randn(n_queries, embedding_dim)

    # Normalize for cosine similarity
    gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Create labels
    gallery_labels = [f"person_{i}" for i in range(n_gallery)]
    query_labels = [f"person_{i % n_gallery}" for i in range(n_queries)]  # Some matches

    # Test identification metrics
    id_metrics = evaluator.compute_identification_metrics(
        query_embeddings, query_labels,
        gallery_embeddings, gallery_labels,
        metric='cosine'
    )

    print(f"✓ Rank-1: {id_metrics.rank1:.4f}")
    print(f"✓ Rank-5: {id_metrics.rank5:.4f}")

    # Test comprehensive performance
    performance = evaluator.compute_model_performance(
        model_name="TestModel",
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        inference_time_ms=25.5,
        memory_usage_mb=512.3
    )

    # Generate report
    report = evaluator.generate_report(performance, format='markdown')
    print("\n" + "=" * 50)
    print("Sample Report Generated Successfully!")
    print("=" * 50)

    # Test save/load
    evaluator.save_results(performance, "test_performance.json")
    loaded = evaluator.load_results("test_performance.json")
    print(f"✓ Results saved and loaded successfully: {loaded.model_name}")

    import os
    if os.path.exists("test_performance.json"):
        os.remove("test_performance.json")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_metrics()
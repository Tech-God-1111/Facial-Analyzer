"""
Evaluation metrics for face recognition research
Comprehensive metrics for verification and identification
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VerificationMetrics:
    """Container for verification metrics"""
    eer: float  # Equal Error Rate
    frr_1_far: float  # False Reject Rate at 1% FAR
    frr_01_far: float  # False Reject Rate at 0.1% FAR
    frr_001_far: float  # False Reject Rate at 0.01% FAR
    auc: float  # Area Under ROC Curve
    tpr_fpr_curve: np.ndarray  # TPR at various FPRs
    thresholds: np.ndarray  # Corresponding thresholds
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


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


class FaceRecognitionEvaluator:
    """
    Comprehensive evaluator for face recognition systems
    Supports both verification and identification tasks
    """

    def __init__(self, num_bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.epsilon = 1e-10  # Small value to avoid division by zero

    def compute_verification_metrics(self,
                                     genuine_scores: np.ndarray,
                                     impostor_scores: np.ndarray,
                                     compute_ci: bool = False) -> VerificationMetrics:
        """
        Compute verification metrics from scores

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

        # Sort scores for threshold sweeping
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

        # Initialize arrays for ROC computation
        frr_list = []  # False Reject Rate (Type I error)
        far_list = []  # False Accept Rate (Type II error)
        tpr_list = []  # True Positive Rate

        for threshold in thresholds:
            # Genuine pairs: predicted negative when threshold > score
            frr = np.mean(genuine_scores < threshold)

            # Impostor pairs: predicted positive when threshold <= score
            far = np.mean(impostor_scores >= threshold)

            # True Positive Rate (1 - FRR)
            tpr = 1.0 - frr

            frr_list.append(frr)
            far_list.append(far)
            tpr_list.append(tpr)

        frr_array = np.array(frr_list)
        far_array = np.array(far_list)
        tpr_array = np.array(tpr_list)

        # Find Equal Error Rate (EER)
        # EER is where FRR = FAR
        eer_idx = np.argmin(np.abs(frr_array - far_array))
        eer = (frr_array[eer_idx] + far_array[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        # Compute AUC using trapezoidal rule
        # Sort by FPR (FAR) for AUC computation
        sorted_indices = np.argsort(far_array)
        sorted_far = far_array[sorted_indices]
        sorted_tpr = tpr_array[sorted_indices]
        auc = np.trapz(sorted_tpr, sorted_far)

        # Find FRR at specific FAR operating points
        def find_frr_at_far(target_far: float) -> float:
            idx = np.argmin(np.abs(far_array - target_far))
            return frr_array[idx]

        frr_1_far = find_frr_at_far(0.01)  # 1% FAR
        frr_01_far = find_frr_at_far(0.001)  # 0.1% FAR
        frr_001_far = find_frr_at_far(0.0001)  # 0.01% FAR

        # TPR at various FPRs (for ROC curve points)
        fpr_points = np.array([0.001, 0.01, 0.1, 0.2, 0.5])
        tpr_at_fpr = np.array([find_frr_at_far(fpr) for fpr in fpr_points])

        # Compute confidence intervals if requested
        confidence_intervals = None
        if compute_ci and len(genuine_scores) > 0 and len(impostor_scores) > 0:
            confidence_intervals = self._compute_bootstrap_ci(
                genuine_scores, impostor_scores
            )

        return VerificationMetrics(
            eer=float(eer),
            frr_1_far=float(frr_1_far),
            frr_01_far=float(frr_01_far),
            frr_001_far=float(frr_001_far),
            auc=float(auc),
            tpr_fpr_curve=np.column_stack([fpr_points, tpr_at_fpr]),
            thresholds=np.column_stack([thresholds, frr_array, far_array]),
            confidence_intervals=confidence_intervals
        )

    def compute_identification_metrics(self,
                                       query_embeddings: np.ndarray,
                                       query_labels: List[str],
                                       gallery_embeddings: np.ndarray,
                                       gallery_labels: List[str],
                                       similarity_matrix: Optional[np.ndarray] = None,
                                       metric: str = 'cosine') -> IdentificationMetrics:
        """
        Compute identification metrics

        Args:
            query_embeddings: Embeddings of probe images
            query_labels: Labels of probe images
            gallery_embeddings: Embeddings of gallery images
            gallery_labels: Labels of gallery images
            similarity_matrix: Pre-computed similarity matrix (optional)
            metric: Distance metric for similarity

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

        for i in range(n_queries):
            query_label = query_labels[i]

            # Get similarities for this query
            similarities = similarity_matrix[i]

            # Sort gallery by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = [gallery_labels[idx] for idx in sorted_indices]

            # Find ranks where correct label appears
            correct_positions = np.where(np.array(sorted_labels) == query_label)[0]

            if len(correct_positions) > 0:
                # Best rank (1-indexed)
                best_rank = correct_positions[0] + 1
                rankings.append(best_rank)

                # Compute Average Precision for this query
                ap = self._compute_average_precision(sorted_labels, query_label)
                average_precisions.append(ap)
            else:
                # Label not found in gallery
                rankings.append(np.inf)
                average_precisions.append(0.0)

        # Compute CMC curve
        max_rank = min(100, n_gallery)  # Limit to reasonable rank
        cmc = np.zeros(max_rank)

        for rank in rankings:
            if rank <= max_rank:
                cmc[int(rank) - 1:] += 1

        cmc = cmc / n_queries

        # Compute rank-k accuracies
        rank1 = np.mean(np.array(rankings) <= 1)
        rank5 = np.mean(np.array(rankings) <= 5)
        rank10 = np.mean(np.array(rankings) <= 10)

        # Precision at different K values
        k_values = [1, 5, 10, 20, 50]
        precision_at_k = {}

        for k in k_values:
            if k <= n_gallery:
                precision_at_k[k] = np.mean(np.array(rankings) <= k)

        # Mean Average Precision
        map_score = np.mean(average_precisions)

        return IdentificationMetrics(
            rank1=float(rank1),
            rank5=float(rank5),
            rank10=float(rank10),
            cmc_curve=cmc,
            precision_at_k=precision_at_k,
            map=float(map_score),
            gallery_size=n_gallery,
            probe_size=n_queries
        )

    def compute_comprehensive_metrics(self,
                                      model_name: str,
                                      genuine_scores: np.ndarray,
                                      impostor_scores: np.ndarray,
                                      query_embeddings: Optional[np.ndarray] = None,
                                      query_labels: Optional[List[str]] = None,
                                      gallery_embeddings: Optional[np.ndarray] = None,
                                      gallery_labels: Optional[List[str]] = None) -> Dict:
        """
        Compute comprehensive metrics for both verification and identification

        Returns:
            Dictionary with all metrics and summary statistics
        """
        results = {
            'model_name': model_name,
            'verification': {},
            'identification': {},
            'statistics': {}
        }

        # Verification metrics
        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            ver_metrics = self.compute_verification_metrics(
                genuine_scores, impostor_scores, compute_ci=True
            )

            results['verification'] = {
                'eer': ver_metrics.eer,
                'auc': ver_metrics.auc,
                'frr_1_far': ver_metrics.frr_1_far,
                'frr_01_far': ver_metrics.frr_01_far,
                'frr_001_far': ver_metrics.frr_001_far,
                'confidence_intervals': ver_metrics.confidence_intervals
            }

        # Identification metrics
        if (query_embeddings is not None and gallery_embeddings is not None and
                query_labels is not None and gallery_labels is not None):
            id_metrics = self.compute_identification_metrics(
                query_embeddings, query_labels,
                gallery_embeddings, gallery_labels
            )

            results['identification'] = {
                'rank1': id_metrics.rank1,
                'rank5': id_metrics.rank5,
                'rank10': id_metrics.rank10,
                'map': id_metrics.map,
                'precision_at_k': id_metrics.precision_at_k,
                'gallery_size': id_metrics.gallery_size,
                'probe_size': id_metrics.probe_size
            }

        # Statistics
        results['statistics'] = {
            'num_genuine_pairs': len(genuine_scores),
            'num_impostor_pairs': len(impostor_scores),
            'genuine_mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
            'genuine_std': float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0,
            'impostor_mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0,
            'impostor_std': float(np.std(impostor_scores)) if len(impostor_scores) > 0 else 0,
            'separation_ratio': self._compute_separation_ratio(genuine_scores, impostor_scores)
        }

        return results

    def _compute_similarity_matrix(self,
                                   query_embeddings: np.ndarray,
                                   gallery_embeddings: np.ndarray,
                                   metric: str = 'cosine') -> np.ndarray:
        """Compute similarity matrix between query and gallery embeddings"""
        n_queries = len(query_embeddings)
        n_gallery = len(gallery_embeddings)

        similarity_matrix = np.zeros((n_queries, n_gallery))

        for i in range(n_queries):
            for j in range(n_gallery):
                if metric == 'cosine':
                    # Cosine similarity
                    sim = np.dot(query_embeddings[i], gallery_embeddings[j])
                    norm_i = np.linalg.norm(query_embeddings[i])
                    norm_j = np.linalg.norm(gallery_embeddings[j])
                    similarity_matrix[i, j] = sim / (norm_i * norm_j + self.epsilon)
                elif metric == 'euclidean':
                    # Euclidean distance (converted to similarity)
                    dist = np.linalg.norm(query_embeddings[i] - gallery_embeddings[j])
                    similarity_matrix[i, j] = 1.0 / (1.0 + dist)
                elif metric == 'l2':
                    # L2 distance (converted to similarity)
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
                relevant_positions.append(i + 1)  # 1-indexed

        if not relevant_positions:
            return 0.0

        # Compute precision at each relevant position
        precisions = []
        for k, pos in enumerate(relevant_positions, 1):
            precision_at_k = k / pos
            precisions.append(precision_at_k)

        # Average precision is mean of precisions
        return np.mean(precisions)

    def _compute_separation_ratio(self,
                                  genuine_scores: np.ndarray,
                                  impostor_scores: np.ndarray) -> float:
        """Compute separation ratio between genuine and impostor distributions"""
        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            return 0.0

        genuine_mean = np.mean(genuine_scores)
        impostor_mean = np.mean(impostor_scores)

        genuine_std = np.std(genuine_scores)
        impostor_std = np.std(impostor_scores)

        # Separation ratio (higher is better)
        mean_diff = abs(genuine_mean - impostor_mean)
        pooled_std = np.sqrt((genuine_std ** 2 + impostor_std ** 2) / 2)

        if pooled_std < self.epsilon:
            return float('inf')

        return float(mean_diff / pooled_std)

    def _compute_bootstrap_ci(self,
                              genuine_scores: np.ndarray,
                              impostor_scores: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics

        Returns:
            Dictionary with confidence intervals for key metrics
        """
        bootstrap_eers = []
        bootstrap_aucs = []
        bootstrap_frr_1_fars = []

        n_genuine = len(genuine_scores)
        n_impostor = len(impostor_scores)

        for _ in range(self.num_bootstrap_samples):
            # Resample with replacement
            gen_resample = np.random.choice(genuine_scores, size=n_genuine, replace=True)
            imp_resample = np.random.choice(impostor_scores, size=n_impostor, replace=True)

            # Compute metrics on resampled data
            try:
                metrics = self.compute_verification_metrics(
                    gen_resample, imp_resample, compute_ci=False
                )
                bootstrap_eers.append(metrics.eer)
                bootstrap_aucs.append(metrics.auc)
                bootstrap_frr_1_fars.append(metrics.frr_1_far)
            except:
                # Skip this bootstrap sample if computation fails
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

    def compute_det_curve(self,
                          genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray,
                          num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DET (Detection Error Tradeoff) curve

        Returns:
            fpr: False Positive Rate (FAR)
            fnr: False Negative Rate (FRR)
            thresholds: Corresponding thresholds
        """
        # Sort all scores
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), num_points)

        fpr_list = []
        fnr_list = []

        for threshold in thresholds:
            # False Positive Rate (FAR)
            fpr = np.mean(impostor_scores >= threshold)

            # False Negative Rate (FRR)
            fnr = np.mean(genuine_scores < threshold)

            fpr_list.append(fpr)
            fnr_list.append(fnr)

        return np.array(fpr_list), np.array(fnr_list), thresholds

    def compute_confidence_metrics(self,
                                   genuine_scores: np.ndarray,
                                   impostor_scores: np.ndarray,
                                   calibration_data: Optional[Tuple] = None) -> Dict:
        """
        Compute confidence calibration metrics

        Returns:
            Dictionary with calibration metrics
        """
        # Combine scores and labels
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([
            np.ones(len(genuine_scores)),  # 1 for genuine
            np.zeros(len(impostor_scores))  # 0 for impostor
        ])

        # Bin the scores
        n_bins = 10
        bins = np.linspace(scores.min(), scores.max(), n_bins + 1)
        bin_indices = np.digitize(scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute calibration curve manually
        prob_true = np.zeros(n_bins)
        prob_pred = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_scores = scores[mask]
                bin_labels = labels[mask]
                prob_true[i] = np.mean(bin_labels)
                prob_pred[i] = np.mean(bin_scores)
                bin_counts[i] = len(bin_scores)

        # Remove empty bins
        valid_bins = bin_counts > 0
        prob_true = prob_true[valid_bins]
        prob_pred = prob_pred[valid_bins]
        bin_counts = bin_counts[valid_bins]

        # Expected Calibration Error (ECE)
        ece = np.sum(bin_counts / len(scores) * np.abs(prob_true - prob_pred))

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0

        # Reliability diagram error
        reliability_error = np.mean((prob_true - prob_pred) ** 2) if len(prob_true) > 0 else 0

        return {
            'ece': float(ece),
            'mce': float(mce),
            'reliability_error': float(reliability_error),
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
        }

    def generate_report(self, metrics_dict: Dict, format: str = 'markdown') -> str:
        """
        Generate evaluation report from metrics

        Args:
            metrics_dict: Dictionary from compute_comprehensive_metrics
            format: 'markdown', 'latex', or 'text'

        Returns:
            Formatted report string
        """
        model_name = metrics_dict.get('model_name', 'Unknown Model')

        if format == 'markdown':
            report = f"# Face Recognition Evaluation Report\n\n"
            report += f"**Model:** {model_name}\n\n"

            report += "## Verification Metrics\n\n"
            ver = metrics_dict.get('verification', {})
            if ver:
                report += f"- **EER:** {ver.get('eer', 0):.4f}\n"
                report += f"- **AUC:** {ver.get('auc', 0):.4f}\n"
                report += f"- **FRR @ 1% FAR:** {ver.get('frr_1_far', 0):.4f}\n"
                report += f"- **FRR @ 0.1% FAR:** {ver.get('frr_01_far', 0):.4f}\n"
                report += f"- **FRR @ 0.01% FAR:** {ver.get('frr_001_far', 0):.4f}\n\n"

            report += "## Identification Metrics\n\n"
            ident = metrics_dict.get('identification', {})
            if ident:
                report += f"- **Rank-1 Accuracy:** {ident.get('rank1', 0):.4f}\n"
                report += f"- **Rank-5 Accuracy:** {ident.get('rank5', 0):.4f}\n"
                report += f"- **Rank-10 Accuracy:** {ident.get('rank10', 0):.4f}\n"
                report += f"- **Mean Average Precision:** {ident.get('map', 0):.4f}\n"
                report += f"- **Gallery Size:** {ident.get('gallery_size', 0)}\n"
                report += f"- **Probe Size:** {ident.get('probe_size', 0)}\n\n"

            report += "## Statistics\n\n"
            stats = metrics_dict.get('statistics', {})
            if stats:
                report += f"- **Genuine Pairs:** {stats.get('num_genuine_pairs', 0)}\n"
                report += f"- **Impostor Pairs:** {stats.get('num_impostor_pairs', 0)}\n"
                report += f"- **Genuine Mean Score:** {stats.get('genuine_mean', 0):.4f}\n"
                report += f"- **Genuine Std Score:** {stats.get('genuine_std', 0):.4f}\n"
                report += f"- **Impostor Mean Score:** {stats.get('impostor_mean', 0):.4f}\n"
                report += f"- **Impostor Std Score:** {stats.get('impostor_std', 0):.4f}\n"
                report += f"- **Separation Ratio:** {stats.get('separation_ratio', 0):.4f}\n"

            return report

        elif format == 'latex':
            # LaTeX formatted report
            report = "\\documentclass{article}\n\\begin{document}\n\n"
            report += f"\\section*{{Face Recognition Evaluation: {model_name}}}\n\n"

            report += "\\subsection*{Verification Metrics}\n"
            ver = metrics_dict.get('verification', {})
            if ver:
                report += f"EER: {ver.get('eer', 0):.4f}\\\\\n"
                report += f"AUC: {ver.get('auc', 0):.4f}\\\\\n"

            report += "\\subsection*{Identification Metrics}\n"
            ident = metrics_dict.get('identification', {})
            if ident:
                report += f"Rank-1: {ident.get('rank1', 0):.4f}\\\\\n"
                report += f"Rank-5: {ident.get('rank5', 0):.4f}\\\\\n"

            report += "\\end{document}"
            return report

        else:  # plain text
            report = f"Face Recognition Evaluation Report\n"
            report += f"Model: {model_name}\n"
            report += "=" * 50 + "\n\n"

            report += "Verification Metrics:\n"
            ver = metrics_dict.get('verification', {})
            if ver:
                report += f"  EER: {ver.get('eer', 0):.4f}\n"
                report += f"  AUC: {ver.get('auc', 0):.4f}\n"

            report += "\nIdentification Metrics:\n"
            ident = metrics_dict.get('identification', {})
            if ident:
                report += f"  Rank-1: {ident.get('rank1', 0):.4f}\n"
                report += f"  Rank-5: {ident.get('rank5', 0):.4f}\n"

            return report

    def compute_optimal_threshold(self,
                                  genuine_scores: np.ndarray,
                                  impostor_scores: np.ndarray,
                                  target_far: Optional[float] = None,
                                  target_frr: Optional[float] = None) -> Dict[str, float]:
        """
        Compute optimal threshold based on target FAR or FRR

        Args:
            genuine_scores: Genuine similarity scores
            impostor_scores: Impostor similarity scores
            target_far: Target False Accept Rate (optional)
            target_frr: Target False Reject Rate (optional)

        Returns:
            Dictionary with optimal threshold and corresponding metrics
        """
        if target_far is None and target_frr is None:
            target_far = 0.01  # Default to 1% FAR

        # Get DET curve
        fpr, fnr, thresholds = self.compute_det_curve(genuine_scores, impostor_scores)

        if target_far is not None:
            # Find threshold for target FAR
            idx = np.argmin(np.abs(fpr - target_far))
            optimal_threshold = thresholds[idx]
            actual_far = fpr[idx]
            actual_frr = fnr[idx]
        else:
            # Find threshold for target FRR
            idx = np.argmin(np.abs(fnr - target_frr))
            optimal_threshold = thresholds[idx]
            actual_far = fpr[idx]
            actual_frr = fnr[idx]

        return {
            'threshold': float(optimal_threshold),
            'far': float(actual_far),
            'frr': float(actual_frr),
            'tpr': float(1 - actual_frr),
            'fnr': float(actual_frr)
        }

    def compute_identification_with_threshold(self,
                                              query_embeddings: np.ndarray,
                                              query_labels: List[str],
                                              gallery_embeddings: np.ndarray,
                                              gallery_labels: List[str],
                                              threshold: float,
                                              metric: str = 'cosine') -> Dict[str, float]:
        """
        Compute identification metrics with a similarity threshold

        Args:
            query_embeddings: Embeddings of probe images
            query_labels: Labels of probe images
            gallery_embeddings: Embeddings of gallery images
            gallery_labels: Labels of gallery images
            threshold: Similarity threshold for matching
            metric: Distance metric for similarity

        Returns:
            Dictionary with identification metrics
        """
        n_queries = len(query_embeddings)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(
            query_embeddings, gallery_embeddings, metric
        )

        tp = 0  # True positives (correct match above threshold)
        fp = 0  # False positives (wrong match above threshold)
        tn = 0  # True negatives (no match below threshold)
        fn = 0  # False negatives (no match above threshold)

        correct_ranks = []

        for i in range(n_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_label = gallery_labels[best_idx]

            if best_similarity >= threshold:
                # Match found
                if best_label == query_label:
                    tp += 1
                    # Find rank of correct match
                    sorted_indices = np.argsort(similarities)[::-1]
                    sorted_labels = [gallery_labels[idx] for idx in sorted_indices]
                    correct_positions = np.where(np.array(sorted_labels) == query_label)[0]
                    if len(correct_positions) > 0:
                        correct_ranks.append(correct_positions[0] + 1)
                else:
                    fp += 1
            else:
                # No match found
                if query_label in gallery_labels:
                    fn += 1
                else:
                    tn += 1

        # Compute metrics
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1_score = 2 * precision * recall / (precision + recall + self.epsilon)

        rank1 = len([r for r in correct_ranks if r == 1]) / (tp + self.epsilon) if len(correct_ranks) > 0 else 0
        rank5 = len([r for r in correct_ranks if r <= 5]) / (tp + self.epsilon) if len(correct_ranks) > 0 else 0

        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'rank1_at_threshold': float(rank1),
            'rank5_at_threshold': float(rank5)
        }


# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Evaluation Metrics...")

    # Create evaluator
    evaluator = FaceRecognitionEvaluator()

    # Generate dummy scores
    np.random.seed(42)
    genuine_scores = np.random.normal(0.8, 0.1, 1000)
    impostor_scores = np.random.normal(0.3, 0.2, 1000)

    # Test verification metrics
    ver_metrics = evaluator.compute_verification_metrics(
        genuine_scores, impostor_scores, compute_ci=False
    )

    print(f"EER: {ver_metrics.eer:.4f}")
    print(f"AUC: {ver_metrics.auc:.4f}")
    print(f"FRR @ 1% FAR: {ver_metrics.frr_1_far:.4f}")

    # Test DET curve
    fpr, fnr, thresholds = evaluator.compute_det_curve(genuine_scores, impostor_scores, num_points=100)
    print(f"DET curve computed: {len(fpr)} points")

    # Test confidence calibration
    calib_metrics = evaluator.compute_confidence_metrics(genuine_scores, impostor_scores)
    print(f"Expected Calibration Error: {calib_metrics['ece']:.4f}")

    # Test optimal threshold
    optimal = evaluator.compute_optimal_threshold(genuine_scores, impostor_scores, target_far=0.01)
    print(f"Optimal threshold for 1% FAR: {optimal['threshold']:.4f}")
    print(f"FRR at optimal threshold: {optimal['frr']:.4f}")

    # Generate dummy embeddings for identification test
    n_gallery = 100
    n_queries = 50
    embedding_dim = 512

    gallery_embeddings = np.random.randn(n_gallery, embedding_dim)
    query_embeddings = np.random.randn(n_queries, embedding_dim)

    # Normalize embeddings (important for cosine similarity)
    gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Create labels (simplified: first 10 queries match gallery)
    gallery_labels = [f"person_{i}" for i in range(n_gallery)]
    query_labels = []
    for i in range(n_queries):
        if i < 10:
            query_labels.append(f"person_{i}")  # Match existing
        else:
            query_labels.append(f"query_{i}")  # New person

    # Test identification metrics
    try:
        id_metrics = evaluator.compute_identification_metrics(
            query_embeddings, query_labels,
            gallery_embeddings, gallery_labels,
            metric='cosine'
        )

        print(f"\nRank-1 Accuracy: {id_metrics.rank1:.4f}")
        print(f"Rank-5 Accuracy: {id_metrics.rank5:.4f}")
        print(f"Mean Average Precision: {id_metrics.map:.4f}")
    except Exception as e:
        print(f"Identification test skipped: {e}")

    # Test comprehensive metrics
    comp_metrics = evaluator.compute_comprehensive_metrics(
        model_name="TestModel",
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels
    )

    # Generate report
    report = evaluator.generate_report(comp_metrics, format='markdown')
    print("\n" + "=" * 50)
    print("Sample Report:")
    print("=" * 50)
    print(report[:500] + "...")  # Print first 500 chars

    print("\n✅ Evaluation metrics testing completed successfully!")
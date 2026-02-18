"""
Face Recognition Benchmarks
Standardized benchmarks for evaluating face recognition models
"""

import numpy as np
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime

from .metrics import FaceRecognitionMetrics, ModelPerformance

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    benchmark_name: str
    model_name: str
    results: Dict[str, Any]
    timestamp: str
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset information"""
    name: str
    description: str
    num_subjects: int
    num_images: int
    image_size: Tuple[int, int]
    protocols: List[str]
    is_public: bool


class FaceRecognitionBenchmark:
    """
    Standardized benchmarks for face recognition evaluation
    """

    def __init__(self):
        self.metrics = FaceRecognitionMetrics()

        # Define standard benchmarks
        self.benchmarks = {
            'verification_small': {
                'description': 'Small-scale verification test',
                'num_genuine': 1000,
                'num_impostor': 10000
            },
            'verification_large': {
                'description': 'Large-scale verification test',
                'num_genuine': 10000,
                'num_impostor': 100000
            },
            'identification_closed': {
                'description': 'Closed-set identification',
                'gallery_size': 1000,
                'probe_size': 100
            },
            'identification_open': {
                'description': 'Open-set identification',
                'gallery_size': 1000,
                'probe_size': 200,
                'unknown_ratio': 0.5
            }
        }

    def run_verification_benchmark(self,
                                   model_name: str,
                                   generate_scores_func,
                                   benchmark_type: str = 'verification_small',
                                   **kwargs) -> BenchmarkResult:
        """
        Run verification benchmark

        Args:
            model_name: Name of the model being evaluated
            generate_scores_func: Function that returns (genuine_scores, impostor_scores)
            benchmark_type: Type of benchmark to run
            **kwargs: Additional arguments for the score generation function

        Returns:
            BenchmarkResult object
        """
        if benchmark_type not in self.benchmarks:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        benchmark_info = self.benchmarks[benchmark_type]
        print(f"Running {benchmark_type}: {benchmark_info['description']}")

        # Generate scores using provided function
        start_time = time.time()
        genuine_scores, impostor_scores = generate_scores_func(**kwargs)
        generation_time = time.time() - start_time

        # Compute metrics
        ver_metrics = self.metrics.compute_verification_metrics(
            genuine_scores, impostor_scores, compute_ci=True
        )

        # Prepare results
        results = {
            'eer': ver_metrics.eer,
            'auc': ver_metrics.auc,
            'frr_1_far': ver_metrics.frr_1_far,
            'frr_01_far': ver_metrics.frr_01_far,
            'frr_001_far': ver_metrics.frr_001_far,
            'd_prime': ver_metrics.d_prime,
            'num_genuine_pairs': len(genuine_scores),
            'num_impostor_pairs': len(impostor_scores),
            'score_generation_time': generation_time,
            'confidence_intervals': ver_metrics.confidence_intervals
        }

        return BenchmarkResult(
            benchmark_name=benchmark_type,
            model_name=model_name,
            results=results,
            timestamp=datetime.now().isoformat(),
            metadata=benchmark_info
        )

    def run_identification_benchmark(self,
                                     model_name: str,
                                     generate_embeddings_func,
                                     benchmark_type: str = 'identification_closed',
                                     **kwargs) -> BenchmarkResult:
        """
        Run identification benchmark

        Args:
            model_name: Name of the model being evaluated
            generate_embeddings_func: Function that returns embeddings and labels
            benchmark_type: Type of benchmark to run
            **kwargs: Additional arguments

        Returns:
            BenchmarkResult object
        """
        if benchmark_type not in self.benchmarks:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        benchmark_info = self.benchmarks[benchmark_type]
        print(f"Running {benchmark_type}: {benchmark_info['description']}")

        # Generate embeddings using provided function
        start_time = time.time()
        embeddings_data = generate_embeddings_func(**kwargs)
        generation_time = time.time() - start_time

        if len(embeddings_data) != 4:
            raise ValueError(
                "Function must return (query_embeddings, query_labels, gallery_embeddings, gallery_labels)")

        query_embeddings, query_labels, gallery_embeddings, gallery_labels = embeddings_data

        # Compute metrics
        open_set = 'open' in benchmark_type
        threshold = 0.5 if open_set else None

        id_metrics = self.metrics.compute_identification_metrics(
            query_embeddings, query_labels,
            gallery_embeddings, gallery_labels,
            metric='cosine',
            open_set=open_set,
            threshold=threshold
        )

        # Prepare results
        results = {
            'rank1': id_metrics.rank1,
            'rank5': id_metrics.rank5,
            'rank10': id_metrics.rank10,
            'map': id_metrics.map,
            'gallery_size': id_metrics.gallery_size,
            'probe_size': id_metrics.probe_size,
            'closed_set': id_metrics.closed_set,
            'hit_rate_at_1_far': id_metrics.hit_rate_at_1_far,
            'embedding_generation_time': generation_time
        }

        return BenchmarkResult(
            benchmark_name=benchmark_type,
            model_name=model_name,
            results=results,
            timestamp=datetime.now().isoformat(),
            metadata=benchmark_info
        )

    def run_comprehensive_benchmark(self,
                                    model_name: str,
                                    generate_scores_func,
                                    generate_embeddings_func,
                                    **kwargs) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark suite

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        # Run verification benchmarks
        print(f"\n{'=' * 60}")
        print(f"Running Comprehensive Benchmarks for {model_name}")
        print(f"{'=' * 60}")

        for bench_name in ['verification_small', 'verification_large']:
            if bench_name in self.benchmarks:
                try:
                    result = self.run_verification_benchmark(
                        model_name=model_name,
                        generate_scores_func=generate_scores_func,
                        benchmark_type=bench_name,
                        **kwargs
                    )
                    results[bench_name] = result
                    print(f"✓ {bench_name}: EER={result.results['eer']:.4f}")
                except Exception as e:
                    print(f"✗ {bench_name} failed: {e}")

        # Run identification benchmarks
        for bench_name in ['identification_closed', 'identification_open']:
            if bench_name in self.benchmarks:
                try:
                    result = self.run_identification_benchmark(
                        model_name=model_name,
                        generate_embeddings_func=generate_embeddings_func,
                        benchmark_type=bench_name,
                        **kwargs
                    )
                    results[bench_name] = result
                    print(f"✓ {bench_name}: Rank-1={result.results['rank1']:.4f}")
                except Exception as e:
                    print(f"✗ {bench_name} failed: {e}")

        return results

    def compare_models(self,
                       model_results: Dict[str, Dict[str, BenchmarkResult]]) -> pd.DataFrame:
        """
        Compare multiple models across benchmarks

        Args:
            model_results: Dictionary of {model_name: {benchmark_name: BenchmarkResult}}

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for model_name, benchmarks in model_results.items():
            for bench_name, result in benchmarks.items():
                row = {
                    'model': model_name,
                    'benchmark': bench_name,
                    'timestamp': result.timestamp
                }

                # Add metrics based on benchmark type
                if 'verification' in bench_name:
                    row.update({
                        'eer': result.results.get('eer', None),
                        'auc': result.results.get('auc', None),
                        'frr_1_far': result.results.get('frr_1_far', None)
                    })
                elif 'identification' in bench_name:
                    row.update({
                        'rank1': result.results.get('rank1', None),
                        'rank5': result.results.get('rank5', None),
                        'map': result.results.get('map', None)
                    })

                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Add summary statistics
        if not df.empty:
            # For verification benchmarks
            ver_df = df[df['benchmark'].str.contains('verification')]
            if not ver_df.empty:
                # Group by model and compute average EER
                avg_eer = ver_df.groupby('model')['eer'].mean()
                df['avg_eer'] = df['model'].map(avg_eer)

            # For identification benchmarks
            id_df = df[df['benchmark'].str.contains('identification')]
            if not id_df.empty:
                avg_rank1 = id_df.groupby('model')['rank1'].mean()
                df['avg_rank1'] = df['model'].map(avg_rank1)

        return df

    def generate_benchmark_report(self,
                                  results: Dict[str, BenchmarkResult],
                                  output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive benchmark report

        Returns:
            Markdown formatted report
        """
        report = "# Face Recognition Benchmark Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary section
        report += "## Summary\n\n"

        # Group by benchmark type
        verification_results = []
        identification_results = []

        for bench_name, result in results.items():
            if 'verification' in bench_name:
                verification_results.append((bench_name, result))
            elif 'identification' in bench_name:
                identification_results.append((bench_name, result))

        # Verification results
        if verification_results:
            report += "### Verification Benchmarks\n\n"
            report += "| Benchmark | EER | AUC | FRR @ 1% FAR |\n"
            report += "|-----------|-----|-----|--------------|\n"

            for bench_name, result in verification_results:
                res = result.results
                report += f"| {bench_name} | {res['eer']:.4f} | {res['auc']:.4f} | {res['frr_1_far']:.4f} |\n"
            report += "\n"

        # Identification results
        if identification_results:
            report += "### Identification Benchmarks\n\n"
            report += "| Benchmark | Rank-1 | Rank-5 | mAP |\n"
            report += "|-----------|--------|--------|-----|\n"

            for bench_name, result in identification_results:
                res = result.results
                report += f"| {bench_name} | {res['rank1']:.4f} | {res['rank5']:.4f} | {res['map']:.4f} |\n"
            report += "\n"

        # Detailed results
        report += "## Detailed Results\n\n"

        for bench_name, result in results.items():
            report += f"### {bench_name}\n\n"
            report += f"**Model:** {result.model_name}\n"
            report += f"**Timestamp:** {result.timestamp}\n\n"

            report += "```json\n"
            report += json.dumps(result.results, indent=2, default=str)
            report += "\n```\n\n"

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")

        return report

    def save_benchmark_results(self,
                               results: BenchmarkResult,
                               output_dir: str = "benchmark_results"):
        """Save benchmark results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results.model_name}_{results.benchmark_name}_{timestamp}.json"
        filepath = output_path / filename

        # Convert to dictionary
        data = {
            'benchmark_name': results.benchmark_name,
            'model_name': results.model_name,
            'timestamp': results.timestamp,
            'results': results.results,
            'metadata': results.metadata
        }

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {filepath}")
        return str(filepath)

    @staticmethod
    def load_benchmark_results(filepath: str) -> BenchmarkResult:
        """Load benchmark results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return BenchmarkResult(**data)


# Example data generators for testing
def generate_dummy_scores(num_genuine: int = 1000, num_impostor: int = 10000):
    """Generate dummy scores for verification testing"""
    np.random.seed(42)

    # Simulate good separation
    genuine_scores = np.random.normal(0.8, 0.1, num_genuine)
    impostor_scores = np.random.normal(0.3, 0.2, num_impostor)

    # Clip to [0, 1]
    genuine_scores = np.clip(genuine_scores, 0, 1)
    impostor_scores = np.clip(impostor_scores, 0, 1)

    return genuine_scores, impostor_scores


def generate_dummy_embeddings(gallery_size: int = 1000, probe_size: int = 100, embedding_dim: int = 512):
    """Generate dummy embeddings for identification testing"""
    np.random.seed(42)

    # Generate gallery embeddings
    gallery_embeddings = np.random.randn(gallery_size, embedding_dim)
    gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    gallery_labels = [f"subject_{i:04d}" for i in range(gallery_size)]

    # Generate probe embeddings (some match gallery, some are unknown)
    query_embeddings = []
    query_labels = []

    # First half match gallery, second half are unknown
    for i in range(probe_size):
        if i < probe_size // 2:
            # Match existing subject
            subject_idx = i % gallery_size
            # Add some noise to make it realistic
            noise = np.random.normal(0, 0.1, embedding_dim)
            embedding = gallery_embeddings[subject_idx] + noise
            embedding = embedding / np.linalg.norm(embedding)
            query_embeddings.append(embedding)
            query_labels.append(gallery_labels[subject_idx])
        else:
            # New subject
            embedding = np.random.randn(embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            query_embeddings.append(embedding)
            query_labels.append(f"unknown_{i}")

    query_embeddings = np.array(query_embeddings)

    return query_embeddings, query_labels, gallery_embeddings, gallery_labels


# Test function
def test_benchmarks():
    """Test the benchmark system"""
    print("Testing Face Recognition Benchmarks...")

    # Create benchmark runner
    benchmark = FaceRecognitionBenchmark()

    # Test verification benchmark
    print("\n1. Testing Verification Benchmark...")
    result = benchmark.run_verification_benchmark(
        model_name="TestModel",
        generate_scores_func=generate_dummy_scores,
        benchmark_type="verification_small"
    )

    print(f"✓ {result.benchmark_name}: EER={result.results['eer']:.4f}")

    # Test identification benchmark
    print("\n2. Testing Identification Benchmark...")
    result = benchmark.run_identification_benchmark(
        model_name="TestModel",
        generate_embeddings_func=generate_dummy_embeddings,
        benchmark_type="identification_closed"
    )

    print(f"✓ {result.benchmark_name}: Rank-1={result.results['rank1']:.4f}")

    # Test comprehensive benchmark
    print("\n3. Testing Comprehensive Benchmark...")
    results = benchmark.run_comprehensive_benchmark(
        model_name="TestModel",
        generate_scores_func=generate_dummy_scores,
        generate_embeddings_func=generate_dummy_embeddings
    )

    print(f"✓ Ran {len(results)} benchmarks")

    # Generate report
    print("\n4. Generating Report...")
    report = benchmark.generate_benchmark_report(results)
    print("✓ Report generated successfully")

    # Test comparison
    print("\n5. Testing Model Comparison...")
    model_results = {
        'ModelA': results,
        'ModelB': results  # Using same results for demo
    }

    comparison_df = benchmark.compare_models(model_results)
    print(f"✓ Compared {len(model_results)} models")
    print(f"Comparison DataFrame shape: {comparison_df.shape}")

    # Test save/load
    print("\n6. Testing Save/Load...")
    for bench_name, result in results.items():
        filepath = benchmark.save_benchmark_results(result)
        loaded = benchmark.load_benchmark_results(filepath)
        print(f"✓ {bench_name} saved and loaded: {loaded.model_name}")

        import os
        if os.path.exists(filepath):
            os.remove(filepath)

    print("\n" + "=" * 60)
    print("✅ All benchmark tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmarks()

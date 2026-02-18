"""
Face Recognition Evaluation Package
"""

from .metrics import (
    FaceRecognitionMetrics,
    VerificationMetrics,
    IdentificationMetrics,
    ModelPerformance
)

from .benchmarks import (
    FaceRecognitionBenchmark,
    BenchmarkResult,
    BenchmarkDataset
)

__all__ = [
    'FaceRecognitionMetrics',
    'VerificationMetrics',
    'IdentificationMetrics',
    'ModelPerformance',
    'FaceRecognitionBenchmark',
    'BenchmarkResult',
    'BenchmarkDataset'
]

__version__ = '1.0.0'
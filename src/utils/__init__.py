"""
Utilities Package for Face Recognition Research
"""

# Import experiment tracker
from .experiment_tracker import (
    ExperimentTracker,
    ExperimentConfig,
    ExperimentResult,
    save_experiment,
    load_experiment
)

# Import data loader
from .data_loader import (
    FaceDatasetLoader,
    DatasetConfig,
    ImageProcessor,
    DataAugmentor,
    BatchGenerator,
    load_image,
    save_image,
    preprocess_face,
    extract_faces
)

# Import visualization
from .visualization import (
    FaceVisualizer,
    MetricsPlotter,
    EmbeddingVisualizer,
    plot_roc_curve,
    plot_cmc_curve,
    plot_confusion_matrix,
    plot_embeddings_tsne,
    plot_embeddings_pca,
    plot_score_distributions,
    plot_verification_pairs,
    plot_identification_results,
    save_figure,
    show_figure,
    create_animation
)

# Import additional utilities (if they exist)
try:
    from .face_utils import (
        align_face,
        detect_landmarks,
        crop_face,
        normalize_face,
        compute_face_quality,
        FaceQualityScore
    )
except ImportError:
    pass

try:
    from .model_utils import (
        load_model,
        save_model,
        ModelWrapper,
        EmbeddingExtractor,
        SimilarityCalculator,
        ThresholdOptimizer
    )
except ImportError:
    pass

try:
    from .file_utils import (
        create_directory,
        list_files,
        get_file_info,
        split_filename,
        generate_unique_id,
        save_json,
        load_json,
        save_pickle,
        load_pickle,
        save_yaml,
        load_yaml
    )
except ImportError:
    pass

try:
    from .time_utils import (
        Timer,
        Profiler,
        measure_performance,
        format_time,
        get_timestamp,
        time_decorator
    )
except ImportError:
    pass

try:
    from .log_utils import (
        setup_logger,
        get_logger,
        log_experiment,
        log_metrics,
        LogManager
    )
except ImportError:
    pass

__all__ = [
    # Experiment Tracker
    'ExperimentTracker',
    'ExperimentConfig',
    'ExperimentResult',
    'save_experiment',
    'load_experiment',

    # Data Loader
    'FaceDatasetLoader',
    'DatasetConfig',
    'ImageProcessor',
    'DataAugmentor',
    'BatchGenerator',
    'load_image',
    'save_image',
    'preprocess_face',
    'extract_faces',

    # Visualization
    'FaceVisualizer',
    'MetricsPlotter',
    'EmbeddingVisualizer',
    'plot_roc_curve',
    'plot_cmc_curve',
    'plot_confusion_matrix',
    'plot_embeddings_tsne',
    'plot_embeddings_pca',
    'plot_score_distributions',
    'plot_verification_pairs',
    'plot_identification_results',
    'save_figure',
    'show_figure',
    'create_animation'
]

__version__ = '1.0.0'

# Convenience functions

def get_all_utils():
    """Get all available utility functions"""
    utils = {}

    # Add all exported functions and classes
    for name in __all__:
        if name in globals():
            utils[name] = globals()[name]

    return utils

def setup_utilities(config: dict = None):
    """
    Setup all utilities with configuration

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of initialized utilities
    """
    config = config or {}

    utilities = {
        'tracker': None,
        'data_loader': None,
        'visualizer': None
    }

    # Initialize experiment tracker
    try:
        tracker_config = config.get('tracker', {})
        utilities['tracker'] = ExperimentTracker(**tracker_config)
    except Exception as e:
        print(f"Warning: Could not initialize ExperimentTracker: {e}")

    # Initialize data loader
    try:
        loader_config = config.get('data_loader', {})
        utilities['data_loader'] = FaceDatasetLoader(**loader_config)
    except Exception as e:
        print(f"Warning: Could not initialize FaceDatasetLoader: {e}")

    # Initialize visualizer
    try:
        vis_config = config.get('visualizer', {})
        utilities['visualizer'] = FaceVisualizer(**vis_config)
    except Exception as e:
        print(f"Warning: Could not initialize FaceVisualizer: {e}")

    return utilities

# Quick test function
def test_utils():
    """Test that all utils modules can be imported"""
    print("Testing Utilities Package...")

    modules = [
        'experiment_tracker',
        'data_loader',
        'visualization'
    ]

    all_ok = True

    for module_name in modules:
        try:
            module = __import__(f'.{module_name}', fromlist=[''])
            print(f"✓ {module_name}: OK")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            all_ok = False

    if all_ok:
        print("\n✅ All utility modules imported successfully!")
    else:
        print("\n⚠ Some utility modules could not be imported")

    return all_ok

if __name__ == "__main__":

    test_utils()

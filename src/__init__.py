"""
Face Recognition Research Package
"""

# Import subpackages
try:
    from . import models
    from . import utils
    from . import evaluation
except ImportError:
    # For direct execution
    pass

__all__ = [
    'models',
    'utils',
    'evaluation'
]

__version__ = '1.0.0'


def get_package_info():
    """Get information about the package"""
    info = {
        'name': 'face_recognition_research',
        'version': __version__,
        'subpackages': ['models', 'utils', 'evaluation'],
        'description': 'Comprehensive face recognition research framework'
    }

    return info


def setup_package(config_path: str = None):
    """
    Setup the complete face recognition research package

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary of initialized components
    """
    import json
    from pathlib import Path

    # Load configuration if provided
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Initialize components
    components = {}

    # Setup utilities
    try:
        utils_config = config.get('utils', {})
        components['utils'] = utils.setup_utilities(utils_config)
    except Exception as e:
        print(f"Warning: Could not setup utilities: {e}")

    # Setup models
    try:
        models_config = config.get('models', {})
        # You could initialize default models here
        components['models'] = {}
    except Exception as e:
        print(f"Warning: Could not setup models: {e}")

    # Setup evaluation
    try:
        eval_config = config.get('evaluation', {})
        components['evaluation'] = {
            'metrics': evaluation.FaceRecognitionMetrics(),
            'benchmark': evaluation.FaceRecognitionBenchmark()
        }
    except Exception as e:
        print(f"Warning: Could not setup evaluation: {e}")

    return components


def test_all():
    """Test all subpackages"""
    print("Testing Face Recognition Research Package...")
    print("=" * 60)

    results = {}

    # Test models
    try:
        from .models.face import test_models
        results['models'] = test_models()
    except Exception as e:
        print(f"Models test failed: {e}")
        results['models'] = False

    # Test utils
    try:
        from .utils import test_utils
        results['utils'] = test_utils()
    except Exception as e:
        print(f"Utils test failed: {e}")
        results['utils'] = False

    # Test evaluation
    try:
        from .evaluation.metrics import test_metrics
        test_metrics()
        from .evaluation.benchmarks import test_benchmarks
        test_benchmarks()
        results['evaluation'] = True
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        results['evaluation'] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Results:")
    for package, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{package:12} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed")

    return all_passed


if __name__ == "__main__":
    test_all()

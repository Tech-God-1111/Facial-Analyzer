"""
Face Recognition Models Package
"""

# Import model implementations
from .facenet import FaceNetModel
from .arcface import ArcFaceModel
from .deepface import DeepFaceModel
from .attention_face import AttentionFaceModel
from .ensemble import EnsembleModel

# Import base classes
try:
    from .base import (
        FaceRecognitionModel,
        EmbeddingModel,
        VerificationModel,
        IdentificationModel,
        ModelFactory
    )
except ImportError:
    pass

# Import model utilities
try:
    from .model_utils import (
        load_pretrained,
        save_model,
        convert_model,
        ModelConfig,
        ModelRegistry
    )
except ImportError:
    pass

__all__ = [
    # Model implementations
    'FaceNetModel',
    'ArcFaceModel',
    'DeepFaceModel',
    'AttentionFaceModel',
    'EnsembleModel',

    # Base classes (if available)
    # 'FaceRecognitionModel',
    # 'EmbeddingModel',
    # 'VerificationModel',
    # 'IdentificationModel',
    # 'ModelFactory'
]

__version__ = '1.0.0'


def get_all_models():
    """Get all available model classes"""
    models = {}

    for name in __all__:
        if name in globals():
            models[name] = globals()[name]

    return models


def create_model(model_name: str, **kwargs):
    """
    Factory function to create a model by name

    Args:
        model_name: Name of the model to create
        **kwargs: Model initialization parameters

    Returns:
        Initialized model instance
    """
    model_classes = {
        'facenet': FaceNetModel,
        'arcface': ArcFaceModel,
        'deepface': DeepFaceModel,
        'attention': AttentionFaceModel,
        'ensemble': EnsembleModel
    }

    model_class = model_classes.get(model_name.lower())
    if model_class is None:
        available = ', '.join(model_classes.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return model_class(**kwargs)


# Quick test function
def test_models():
    """Test that all model modules can be imported"""
    print("Testing Face Models Package...")

    modules = [
        'facenet',
        'arcface',
        'deepface',
        'attention_face',
        'ensemble'
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
        print("\n✅ All model modules imported successfully!")
    else:
        print("\n⚠ Some model modules could not be imported")

    return all_ok


if __name__ == "__main__":

    test_models()

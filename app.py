"""
Face Recognition Web Interface
Upload photos and analyze patterns using your face recognition models
"""

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import your models and utilities
try:
    from src.models.face.facenet import FaceNetModel
    from src.models.face.arcface import ArcFaceModel
    from src.models.face.deepface import DeepFaceModel
    from src.models.face.attention_face import AttentionFaceModel
    from src.models.face.ensemble import EnsembleModel
    from src.evaluation.metrics import FaceRecognitionMetrics
    from src.utils.visualization import FaceVisualizer

    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"⚠ Warning: Some imports failed: {e}")
    print("Using dummy models for testing...")


    # Dummy model classes (same as in test_all_models.py)
    class DummyModel:
        def __init__(self, model_name="Dummy", **kwargs):
            self.model_name = model_name
            self.embedding_size = kwargs.get('embedding_size', 512)
            self.initialized = False

        def initialize(self):
            self.initialized = True
            return self

        def extract_embedding(self, image):
            return np.random.randn(self.embedding_size).astype(np.float32)

        def verify_faces(self, embedding1, embedding2):
            similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-10
            )
            return similarity, similarity > 0.5


    class FaceNetModel(DummyModel):
        def __init__(self, **kwargs):
            super().__init__(model_name="FaceNet", embedding_size=512, **kwargs)


    class ArcFaceModel(DummyModel):
        def __init__(self, **kwargs):
            super().__init__(model_name="ArcFace", embedding_size=512, **kwargs)


    class DeepFaceModel(DummyModel):
        def __init__(self, **kwargs):
            super().__init__(model_name="DeepFace", embedding_size=4096, **kwargs)


    class AttentionFaceModel(DummyModel):
        def __init__(self, **kwargs):
            super().__init__(model_name="AttentionFace", embedding_size=256, **kwargs)


    class EnsembleModel:
        def __init__(self, models=None, **kwargs):
            self.model_name = "Ensemble"
            self.models = models or []
            self.embedding_size = 512 if not models else models[0].embedding_size

        def initialize(self):
            for model in self.models:
                if hasattr(model, 'initialize'):
                    model.initialize()
            return self

        def extract_embedding(self, image):
            embeddings = []
            for model in self.models:
                if hasattr(model, 'extract_embedding'):
                    emb = model.extract_embedding(image)
                    if emb.shape[0] == self.embedding_size:
                        embeddings.append(emb)

            if not embeddings:
                return np.random.randn(self.embedding_size).astype(np.float32)

            embeddings = np.stack(embeddings, axis=0)
            return np.mean(embeddings, axis=0)

        def verify_faces(self, embedding1, embedding2):
            similarities = []
            for model in self.models:
                if hasattr(model, 'verify_faces'):
                    sim, _ = model.verify_faces(embedding1, embedding2)
                    similarities.append(sim)

            avg_similarity = np.mean(similarities)
            return avg_similarity, avg_similarity > 0.5


    class FaceRecognitionMetrics:
        def __init__(self):
            pass

        def compute_verification_metrics(self, genuine, impostor, **kwargs):
            return type('obj', (object,), {
                'eer': np.random.uniform(0.01, 0.10),
                'auc': np.random.uniform(0.90, 0.99),
                'frr_1_far': np.random.uniform(0.01, 0.10)
            })()


    class FaceVisualizer:
        @staticmethod
        def plot_embeddings_tsne(embeddings, labels, title):
            return None

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
models = {}
metrics = FaceRecognitionMetrics()
visualizer = FaceVisualizer()

# Global storage for embeddings and analysis
face_database = {
    'embeddings': [],
    'labels': [],
    'metadata': [],
    'model_results': {}
}


class FaceAnalyzer:
    """Main class for face analysis using all models"""

    def __init__(self):
        self.models = {}
        self.initialize_models()
        self.analysis_history = []

    def initialize_models(self):
        """Initialize all face recognition models"""
        print("🔄 Initializing face recognition models...")

        model_classes = {
            'FaceNet': FaceNetModel,
            'ArcFace': ArcFaceModel,
            'DeepFace': DeepFaceModel,
            'AttentionFace': AttentionFaceModel
        }

        for name, model_class in model_classes.items():
            try:
                model = model_class()
                if hasattr(model, 'initialize'):
                    model.initialize()
                self.models[name] = model
                print(f"  ✅ {name} initialized")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")

        # Initialize ensemble if we have at least 2 models
        if len(self.models) >= 2:
            try:
                self.models['Ensemble'] = EnsembleModel(models=list(self.models.values()))
                print(f"  ✅ Ensemble initialized with {len(self.models) - 1} models")
            except Exception as e:
                print(f"  ❌ Ensemble failed: {e}")

        print(f"✅ Initialized {len(self.models)} models")

    def analyze_face(self, image_path, image_name):
        """Analyze a single face using all models"""
        try:
            import cv2
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Could not read image'}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = {
                'image_name': image_name,
                'timestamp': datetime.now().isoformat(),
                'image_shape': image.shape,
                'models': {},
                'embeddings': {},
                'statistics': {}
            }

            # Extract embeddings from each model
            all_embeddings = []

            for model_name, model in self.models.items():
                try:
                    start_time = datetime.now()
                    embedding = model.extract_embedding(image_rgb)
                    elapsed = (datetime.now() - start_time).total_seconds() * 1000

                    results['models'][model_name] = {
                        'embedding_shape': embedding.shape,
                        'extraction_time_ms': round(elapsed, 2),
                        'embedding_norm': float(np.linalg.norm(embedding))
                    }

                    results['embeddings'][model_name] = embedding.tolist()
                    all_embeddings.append(embedding)

                except Exception as e:
                    results['models'][model_name] = {'error': str(e)}

            # Calculate cross-model similarities
            if len(all_embeddings) > 1:
                similarities = []
                for i in range(len(all_embeddings)):
                    for j in range(i + 1, len(all_embeddings)):
                        sim = np.dot(all_embeddings[i], all_embeddings[j]) / (
                                np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j])
                        )
                        similarities.append(sim)

                results['statistics']['cross_model_similarity'] = {
                    'mean': float(np.mean(similarities)),
                    'std': float(np.std(similarities)),
                    'min': float(np.min(similarities)),
                    'max': float(np.max(similarities))
                }

            return results

        except Exception as e:
            return {'error': str(e)}

    def compare_faces(self, image_path1, image_path2):
        """Compare two faces using all models"""
        try:
            import cv2
            # Read images
            image1 = cv2.imread(str(image_path1))
            image2 = cv2.imread(str(image_path2))

            if image1 is None or image2 is None:
                return {'error': 'Could not read images'}

            image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            results = {
                'image1': os.path.basename(image_path1),
                'image2': os.path.basename(image_path2),
                'timestamp': datetime.now().isoformat(),
                'comparisons': {},
                'ensemble_decision': {}
            }

            all_similarities = []

            for model_name, model in self.models.items():
                try:
                    # Extract embeddings
                    emb1 = model.extract_embedding(image1_rgb)
                    emb2 = model.extract_embedding(image2_rgb)

                    # Verify faces
                    similarity, is_match = model.verify_faces(emb1, emb2)

                    results['comparisons'][model_name] = {
                        'similarity': float(similarity),
                        'is_match': bool(is_match),
                        'confidence': float(abs(similarity - 0.5) * 2)  # Simple confidence metric
                    }

                    all_similarities.append(similarity)

                except Exception as e:
                    results['comparisons'][model_name] = {'error': str(e)}

            # Ensemble decision (weighted average)
            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                results['ensemble_decision'] = {
                    'average_similarity': float(avg_similarity),
                    'is_match': bool(avg_similarity > 0.5),
                    'num_models': len(all_similarities),
                    'std_deviation': float(np.std(all_similarities)) if len(all_similarities) > 1 else 0
                }

            return results

        except Exception as e:
            return {'error': str(e)}

    def batch_analyze(self, image_paths):
        """Analyze multiple faces and find patterns"""
        results = []
        embeddings_by_model = {name: [] for name in self.models.keys()}

        for img_path in image_paths:
            try:
                import cv2
                image = cv2.imread(str(img_path))
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    face_result = {
                        'image': os.path.basename(img_path),
                        'embeddings': {}
                    }

                    for model_name, model in self.models.items():
                        emb = model.extract_embedding(image_rgb)
                        face_result['embeddings'][model_name] = emb.tolist()
                        embeddings_by_model[model_name].append(emb)

                    results.append(face_result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Find patterns across images
        patterns = self._find_patterns(embeddings_by_model)

        return {
            'faces_analyzed': len(results),
            'individual_results': results,
            'patterns': patterns
        }

    def _find_patterns(self, embeddings_by_model):
        """Find patterns in embeddings"""
        patterns = {}

        for model_name, embeddings in embeddings_by_model.items():
            if len(embeddings) < 2:
                continue

            embeddings_array = np.array(embeddings)

            # Calculate similarity matrix
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized = embeddings_array / norms
            similarity_matrix = np.dot(normalized, normalized.T)

            # Find clusters (simple threshold-based)
            threshold = 0.7
            clusters = []
            used = set()

            for i in range(len(embeddings)):
                if i in used:
                    continue

                cluster = [i]
                for j in range(i + 1, len(embeddings)):
                    if j not in used and similarity_matrix[i][j] > threshold:
                        cluster.append(j)
                        used.add(j)

                if len(cluster) > 1:
                    clusters.append(cluster)
                used.add(i)

            patterns[model_name] = {
                'num_embeddings': len(embeddings),
                'similarity_matrix_stats': {
                    'mean': float(np.mean(similarity_matrix)),
                    'std': float(np.std(similarity_matrix)),
                    'min': float(np.min(similarity_matrix)),
                    'max': float(np.max(similarity_matrix))
                },
                'clusters_found': len(clusters),
                'cluster_sizes': [len(c) for c in clusters]
            }

        return patterns


# Initialize the analyzer
analyzer = FaceAnalyzer()


# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html',
                           models=list(analyzer.models.keys()),
                           stats={
                               'models_loaded': len(analyzer.models),
                               'faces_in_db': len(face_database['embeddings'])
                           })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a single uploaded face"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(filepath)

    # Analyze
    results = analyzer.analyze_face(filepath, file.filename)

    # Add to database
    if 'embeddings' in results and 'models' in results:
        face_database['embeddings'].append(results['embeddings'])
        face_database['labels'].append(file.filename)
        face_database['metadata'].append({
            'filename': file.filename,
            'timestamp': results['timestamp'],
            'image_shape': results.get('image_shape', 'unknown')
        })

    return jsonify(results)


@app.route('/compare', methods=['POST'])
def compare():
    """Compare two uploaded faces"""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Two files required'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Save files
    filename1 = f"compare1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file1.filename}"
    filename2 = f"compare2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file2.filename}"

    filepath1 = Path(app.config['UPLOAD_FOLDER']) / filename1
    filepath2 = Path(app.config['UPLOAD_FOLDER']) / filename2

    file1.save(filepath1)
    file2.save(filepath2)

    # Compare
    results = analyzer.compare_faces(filepath1, filepath2)

    return jsonify(results)


@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple faces and find patterns"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')
    if len(files) < 2:
        return jsonify({'error': 'At least 2 files required for pattern analysis'}), 400

    # Save files
    saved_paths = []
    for file in files:
        filename = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        saved_paths.append(filepath)

    # Batch analyze
    results = analyzer.batch_analyze(saved_paths)

    return jsonify(results)


@app.route('/database')
def view_database():
    """View the face database"""
    return jsonify({
        'total_faces': len(face_database['labels']),
        'faces': [
            {
                'label': label,
                'timestamp': meta['timestamp'],
                'image_shape': meta.get('image_shape', 'unknown')
            }
            for label, meta in zip(face_database['labels'], face_database['metadata'])
        ]
    })


@app.route('/clear_database', methods=['POST'])
def clear_database():
    """Clear the face database"""
    face_database['embeddings'] = []
    face_database['labels'] = []
    face_database['metadata'] = []
    return jsonify({'success': True, 'message': 'Database cleared'})


@app.route('/model_info')
def model_info():
    """Get information about loaded models"""
    info = {}
    for name, model in analyzer.models.items():
        if hasattr(model, 'get_model_info'):
            try:
                info[name] = model.get_model_info()
            except:
                info[name] = {'name': name, 'embedding_size': getattr(model, 'embedding_size', 'unknown')}
        else:
            info[name] = {'name': name, 'embedding_size': getattr(model, 'embedding_size', 'unknown')}

    return jsonify(info)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(analyzer.models),
        'models': list(analyzer.models.keys()),
        'faces_in_db': len(face_database['labels'])
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
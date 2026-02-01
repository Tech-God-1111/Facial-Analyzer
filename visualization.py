"""
Visualization utilities for face recognition research
Create publication-quality figures and plots
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import cv2
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap


class FaceVisualizer:
    """
    Visualization tools for face recognition analysis
    """

    def __init__(self, style: str = "seaborn"):
        """Initialize visualizer with style"""
        if style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.2)
        elif style == "matplotlib":
            plt.style.use('default')

        self.figsize = (10, 8)
        self.dpi = 300
        self.color_palette = sns.color_palette("husl", 8)

        # Custom colormap for attention
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['blue', 'cyan', 'yellow', 'red']
        )

    def plot_face_grid(self,
                       faces: List[np.ndarray],
                       titles: List[str] = None,
                       n_cols: int = 4,
                       figsize: Tuple[int, int] = None):
        """
        Plot grid of face images
        """
        n_faces = len(faces)
        n_rows = (n_faces + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i in range(n_rows * n_cols):
            if i < n_faces:
                # Convert BGR to RGB for display
                if len(faces[i].shape) == 3 and faces[i].shape[2] == 3:
                    display_img = cv2.cvtColor(faces[i], cv2.COLOR_BGR2RGB)
                else:
                    display_img = faces[i]

                axes[i].imshow(display_img)
                if titles and i < len(titles):
                    axes[i].set_title(titles[i], fontsize=10)
                axes[i].axis('off')
            else:
                axes[i].axis('off')

        plt.tight_layout()
        return fig

    def plot_face_with_bbox(self,
                            face: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            landmarks: List[Tuple[int, int]] = None,
                            title: str = "Face Detection"):
        """
        Plot face with bounding box and landmarks
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Convert BGR to RGB
        display_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)

        # Draw bounding box
        x, y, w, h = bbox
        rect = Rectangle((x, y), w, h,
                         linewidth=2,
                         edgecolor='lime',
                         facecolor='none')
        ax.add_patch(rect)

        # Draw landmarks if provided
        if landmarks:
            colors = ['red', 'green', 'blue', 'cyan', 'magenta']
            for i, (lx, ly) in enumerate(landmarks):
                color = colors[i % len(colors)]
                circle = Circle((lx, ly), radius=3,
                                color=color,
                                fill=True)
                ax.add_patch(circle)
                ax.text(lx, ly - 10, f'P{i}',
                        color=color, fontsize=8,
                        ha='center', va='bottom')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig

    def plot_attention_overlay(self,
                               face: np.ndarray,
                               attention_map: np.ndarray,
                               title: str = "Attention Visualization"):
        """
        Plot face with attention overlay
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original face
        display_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        axes[0].imshow(display_face)
        axes[0].set_title("Original Face", fontsize=12)
        axes[0].axis('off')

        # Attention map
        if attention_map.ndim == 2:
            im = axes[1].imshow(attention_map, cmap=self.attention_cmap)
            axes[1].set_title("Attention Map", fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        elif attention_map.ndim == 3:
            # Take mean across channels
            mean_attention = attention_map.mean(axis=0)
            im = axes[1].imshow(mean_attention, cmap=self.attention_cmap)
            axes[1].set_title("Mean Attention", fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(display_face)

        # Resize attention map to match face dimensions
        if attention_map.ndim == 2:
            resized_attention = cv2.resize(attention_map,
                                           (face.shape[1], face.shape[0]))
            axes[2].imshow(resized_attention,
                           cmap=self.attention_cmap,
                           alpha=0.5)
        elif attention_map.ndim == 3:
            resized_attention = cv2.resize(attention_map.mean(axis=0),
                                           (face.shape[1], face.shape[0]))
            axes[2].imshow(resized_attention,
                           cmap=self.attention_cmap,
                           alpha=0.5)

        axes[2].set_title("Attention Overlay", fontsize=12)
        axes[2].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_embedding_space(self,
                             embeddings: np.ndarray,
                             labels: List[str],
                             method: str = "tsne",
                             title: str = "Embedding Space Visualization",
                             figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize embeddings in 2D space
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2,
                           random_state=42,
                           perplexity=min(30, len(embeddings) - 1))
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")

        embeddings_2d = reducer.fit_transform(embeddings)

        # Create color mapping for labels
        unique_labels = list(set(labels))
        label_to_color = {label: self.color_palette[i % len(self.color_palette)]
                          for i, label in enumerate(unique_labels)}

        # Plot each label
        for label in unique_labels:
            mask = np.array([l == label for l in labels])
            if mask.any():
                ax.scatter(embeddings_2d[mask, 0],
                           embeddings_2d[mask, 1],
                           color=label_to_color[label],
                           label=label,
                           alpha=0.7,
                           s=50,
                           edgecolor='black',
                           linewidth=0.5)

        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add legend (limit to 10 labels for clarity)
        if len(unique_labels) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Show legend with most frequent labels
            label_counts = pd.Series(labels).value_counts()
            top_labels = label_counts.head(5).index.tolist()

            handles = []
            for label in top_labels:
                handles.append(plt.Line2D([0], [0],
                                          marker='o',
                                          color='w',
                                          markerfacecolor=label_to_color[label],
                                          markersize=10,
                                          label=f"{label} ({label_counts[label]})"))

            ax.legend(handles=handles,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      title=f"Top 5 labels (of {len(unique_labels)})")

        plt.tight_layout()
        return fig

    def plot_similarity_matrix(self,
                               similarity_matrix: np.ndarray,
                               labels: List[str] = None,
                               title: str = "Face Similarity Matrix",
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Plot similarity matrix as heatmap
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Create heatmap
        im = ax.imshow(similarity_matrix,
                       cmap='viridis',
                       vmin=0,
                       vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity Score', fontsize=12)

        # Add labels if provided
        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Face Index', fontsize=12)
        ax.set_ylabel('Face Index', fontsize=12)

        # Add text annotations for high similarity
        threshold = 0.8
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                if i != j and similarity_matrix[i, j] > threshold:
                    ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color='white', fontsize=6,
                            fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_roc_curve(self,
                       fpr: np.ndarray,
                       tpr: np.ndarray,
                       roc_auc: float,
                       title: str = "ROC Curve",
                       figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve for face verification
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot ROC curve
        ax.plot(fpr, tpr,
                color='darkorange',
                lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1],
                color='navy',
                lw=2,
                linestyle='--',
                label='Random')

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add inset for zoomed region
        if roc_auc > 0.9:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            ax_inset = inset_axes(ax,
                                  width="40%",
                                  height="40%",
                                  loc='upper left')
            ax_inset.plot(fpr, tpr, color='darkorange', lw=2)
            ax_inset.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_inset.set_xlim([0.0, 0.2])
            ax_inset.set_ylim([0.8, 1.0])
            ax_inset.set_xlabel('FPR', fontsize=8)
            ax_inset.set_ylabel('TPR', fontsize=8)
            ax_inset.set_title('Zoomed Region', fontsize=9)
            ax_inset.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(self,
                                    precision: np.ndarray,
                                    recall: np.ndarray,
                                    average_precision: float,
                                    title: str = "Precision-Recall Curve",
                                    figsize: Tuple[int, int] = (8, 6)):
        """
        Plot precision-recall curve
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(recall, precision,
                color='darkgreen',
                lw=2,
                label=f'PR curve (AP = {average_precision:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_histogram_comparison(self,
                                  genuine_scores: np.ndarray,
                                  impostor_scores: np.ndarray,
                                  threshold: float = None,
                                  title: str = "Score Distribution",
                                  figsize: Tuple[int, int] = (10, 6)):
        """
        Plot histogram comparison of genuine and impostor scores
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot histograms
        ax.hist(genuine_scores,
                bins=50,
                alpha=0.7,
                color='green',
                label=f'Genuine (n={len(genuine_scores)})',
                density=True)

        ax.hist(impostor_scores,
                bins=50,
                alpha=0.7,
                color='red',
                label=f'Impostor (n={len(impostor_scores)})',
                density=True)

        # Add threshold line if provided
        if threshold is not None:
            ax.axvline(x=threshold,
                       color='blue',
                       linestyle='--',
                       linewidth=2,
                       label=f'Threshold = {threshold:.3f}')

        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (f'Genuine: μ={np.mean(genuine_scores):.3f}, '
                      f'σ={np.std(genuine_scores):.3f}\n'
                      f'Impostor: μ={np.mean(impostor_scores):.3f}, '
                      f'σ={np.std(impostor_scores):.3f}')

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round',
                          facecolor='wheat',
                          alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_model_comparison(self,
                              model_results: Dict[str, Dict],
                              metric: str = 'accuracy',
                              title: str = "Model Comparison",
                              figsize: Tuple[int, int] = (12, 6)):
        """
        Plot comparison of multiple models
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Extract data
        model_names = list(model_results.keys())
        metrics_data = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1']}

        for model_name, results in model_results.items():
            for metric_name in metrics_data.keys():
                if metric_name in results:
                    metrics_data[metric_name].append(results[metric_name])

        # Bar plot for specified metric
        if metric in metrics_data and metrics_data[metric]:
            x = np.arange(len(model_names))
            bars = axes[0].bar(x, metrics_data[metric],
                               color=self.color_palette[:len(model_names)])

            axes[0].set_xlabel('Model', fontsize=12)
            axes[0].set_ylabel(metric.capitalize(), fontsize=12)
            axes[0].set_title(f'{metric.capitalize()} Comparison', fontsize=13)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Radar chart for all metrics
        if all(len(v) == len(model_names) for v in metrics_data.values()):
            # Prepare data for radar chart
            metrics_list = list(metrics_data.keys())
            num_vars = len(metrics_list)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # Close the circle

            # Radar chart
            ax_radar = axes[1]
            ax_radar.set_theta_offset(np.pi / 2)
            ax_radar.set_theta_direction(-1)

            # Draw axis lines
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels([m.capitalize() for m in metrics_list], fontsize=10)

            # Draw ylabels
            ax_radar.set_rlabel_position(0)
            ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
            ax_radar.set_ylim(0, 1)

            # Plot each model
            for i, model_name in enumerate(model_names):
                values = [model_results[model_name].get(m, 0) for m in metrics_list]
                values += values[:1]  # Close the circle

                ax_radar.plot(angles, values, linewidth=2,
                              label=model_name,
                              color=self.color_palette[i])
                ax_radar.fill(angles, values,
                              alpha=0.1,
                              color=self.color_palette[i])

            ax_radar.set_title('Metrics Radar Chart', fontsize=13)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

        else:
            axes[1].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_training_history(self,
                              history: Dict[str, List[float]],
                              title: str = "Training History",
                              figsize: Tuple[int, int] = (12, 5)):
        """
        Plot training and validation metrics over epochs
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'],
                         label='Training Loss',
                         color='blue',
                         linewidth=2)

        if 'val_loss' in history:
            axes[0].plot(history['val_loss'],
                         label='Validation Loss',
                         color='red',
                         linewidth=2,
                         linestyle='--')

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss Curve', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        if 'train_accuracy' in history:
            axes[1].plot(history['train_accuracy'],
                         label='Training Accuracy',
                         color='green',
                         linewidth=2)

        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'],
                         label='Validation Accuracy',
                         color='orange',
                         linewidth=2,
                         linestyle='--')

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy Curve', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def save_figure(self, fig, filename: str, dpi: int = None):
        """
        Save figure to file
        """
        if dpi is None:
            dpi = self.dpi

        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"💾 Figure saved: {filename}")


# Example usage and testing
if __name__ == "__main__":
    print("🎨 Testing Visualization Utilities...")

    # Create visualizer
    viz = FaceVisualizer(style="seaborn")

    # Test with dummy data
    dummy_faces = []
    for i in range(6):
        face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        dummy_faces.append(face)

    # Test face grid
    fig1 = viz.plot_face_grid(dummy_faces,
                              titles=[f"Face {i}" for i in range(6)],
                              n_cols=3)
    fig1.suptitle("Test: Face Grid", fontsize=16, fontweight='bold')

    # Test embedding visualization
    dummy_embeddings = np.random.randn(100, 512)
    dummy_labels = [f"Label_{i % 5}" for i in range(100)]

    fig2 = viz.plot_embedding_space(dummy_embeddings, dummy_labels, method="pca")

    # Test ROC curve
    dummy_fpr = np.linspace(0, 1, 100)
    dummy_tpr = np.sin(dummy_fpr * np.pi / 2)  # Simulated ROC
    roc_auc = np.trapz(dummy_tpr, dummy_fpr)

    fig3 = viz.plot_roc_curve(dummy_fpr, dummy_tpr, roc_auc)

    # Save test figures
    viz.save_figure(fig1, "test_face_grid.png")
    viz.save_figure(fig2, "test_embedding_space.png")
    viz.save_figure(fig3, "test_roc_curve.png")

    print("\n✅ Visualization utilities ready!")
    print("\nAvailable visualizations:")
    print("1. plot_face_grid() - Grid of face images")
    print("2. plot_face_with_bbox() - Face with bounding box")
    print("3. plot_attention_overlay() - Attention visualization")
    print("4. plot_embedding_space() - 2D embedding visualization")
    print("5. plot_similarity_matrix() - Similarity heatmap")
    print("6. plot_roc_curve() - ROC curve")
    print("7. plot_precision_recall_curve() - PR curve")
    print("8. plot_histogram_comparison() - Score distributions")
    print("9. plot_model_comparison() - Model comparison")
    print("10. plot_training_history() - Training curves")

    # Close figures
    plt.close('all')
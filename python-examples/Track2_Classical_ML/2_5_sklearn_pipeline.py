"""
Track 2.5: Scikit-learn in Practice
====================================
Complete ML pipelines using Scikit-learn:
- Load data, preprocess, train, evaluate
- Classification, regression, clustering examples

Author: AI Engineering Masterclass
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes, make_blobs, make_moons
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    r2_score, silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# EXAMPLE 1: CLASSIFICATION PIPELINE (IRIS DATASET)
# ==============================================================================

def iris_classification():
    """Complete classification pipeline with multiple models."""

    print("=" * 70)
    print("  EXAMPLE 1: IRIS CLASSIFICATION")
    print("=" * 70)

    # Step 1: Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Features: {list(feature_names)}")
    print(f"Classes: {list(target_names)}")

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Step 3: Preprocess - scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nAfter scaling (mean ≈ 0, std ≈ 1):")
    print(f"  X_train mean: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"  X_train std:  {X_train_scaled.std(axis=0).round(2)}")

    # Step 4: Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }

    print("\n" + "-" * 50)
    print("Model Comparison:")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        # Cross-validation for reliable estimate
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model.fit(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test': test_score
        }
        print(f"\n{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test Accuracy: {test_score:.3f}")

    # Step 5: Detailed evaluation of best model
    best_model = max(results, key=lambda k: results[k]['test'])
    print(f"\n\nBest Model: {best_model}")

    best_clf = models[best_model]
    y_pred = best_clf.predict(X_test_scaled)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("          Predicted")
    print("          setosa versi virginica")
    for i, name in enumerate(target_names):
        print(f"Actual {name:10s} [{cm[i].tolist()}]")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return results

# ==============================================================================
# EXAMPLE 2: REGRESSION PIPELINE (DIABETES DATASET)
# ==============================================================================

def diabetes_regression():
    """Complete regression pipeline with model selection."""

    print("\n" + "=" * 70)
    print("  EXAMPLE 2: DIABETES PROGRESSION REGRESSION")
    print("=" * 70)

    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names

    print(f"\nDataset: Diabetes ({X.shape[0]} samples, {X.shape[1]} features)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Ridge (α=10.0)': Ridge(alpha=10.0)
    }

    print("\n" + "-" * 50)
    print("Regularization Effect (Ridge vs Linear):")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }

        print(f"\n{name}:")
        print(f"  Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
        print(f"  Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    print("\nNote: Linear Regression has NO regularization.")
    print("Ridge with high α = strong regularization → simpler model")

    # Grid search for best alpha
    alphas = np.logspace(-3, 3, 50)
    train_scores = []
    test_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        train_scores.append(r2_score(y_train, model.predict(X_train_scaled)))
        test_scores.append(r2_score(y_test, model.predict(X_test_scaled)))

    best_alpha = alphas[np.argmax(test_scores)]
    print(f"\nOptimal α via Grid Search: {best_alpha:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: R² vs Alpha
    ax1 = axes[0]
    ax1.semilogx(alphas, train_scores, 'b-', label='Train R²')
    ax1.semilogx(alphas, test_scores, 'r-', label='Test R²')
    ax1.axvline(x=best_alpha, color='green', linestyle='--', label=f'Best α = {best_alpha:.2f}')
    ax1.set_xlabel('Alpha (regularization strength)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Ridge Regression: Finding Optimal Regularization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted vs Actual
    ax2 = axes[1]
    best_model = Ridge(alpha=best_alpha)
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    ax2.scatter(y_test, y_pred, alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Diabetes Progression')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Predictions vs Actual (R² = {r2_score(y_test, y_pred):.3f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_pipeline.png', dpi=150)
    plt.show()
    print("Saved: regression_pipeline.png")

    return results

# ==============================================================================
# EXAMPLE 3: CLUSTERING (UNSUPERVISED)
# ==============================================================================

def clustering_example():
    """K-Means and DBSCAN clustering."""

    print("\n" + "=" * 70)
    print("  EXAMPLE 3: CLUSTERING (UNSUPERVISED)")
    print("=" * 70)

    # Generate data with 3 clusters
    X, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.6,
                                random_state=42)

    print(f"\nGenerated {X.shape[0]} points with 3 clusters")

    # K-Means with different k
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, k in enumerate([2, 3, 4, 5, 6, 7]):
        ax = axes[idx // 3, idx % 3]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        colors = plt.cm.Set1(np.linspace(0, 1, k))
        for i in range(k):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.6, s=30)

        # Centroids
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='black', marker='X', s=200, edgecolors='white', linewidths=2)

        # Silhouette score
        from sklearn.metrics import silhouette_score
        if k >= 2:
            sil_score = silhouette_score(X, labels)
            ax.set_title(f'K={k}, Silhouette={sil_score:.3f}')
        else:
            ax.set_title(f'K={k}')

        ax.grid(True, alpha=0.3)

    plt.suptitle('K-Means Clustering: Effect of K', fontsize=14)
    plt.tight_layout()
    plt.savefig('kmeans_analysis.png', dpi=150)
    plt.show()
    print("Saved: kmeans_analysis.png")

    # DBSCAN demo (density-based)
    print("\n" + "-" * 50)
    print("DBSCAN (Density-Based Clustering):")
    print("-" * 50)

    # Create non-spherical clusters
    X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # K-Means struggles with non-spherical clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_moons)

    axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans, cmap='Set1')
    axes[0].set_title('K-Means: Fails on Non-Spherical Clusters')
    axes[0].grid(True, alpha=0.3)

    # DBSCAN with different eps
    for idx, eps in enumerate([0.1, 0.3, 0.5]):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X_moons)
        n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        n_noise = list(labels_dbscan).count(-1)

        axes[idx + 1].scatter(X_moons[:, 0], X_moons[:, 1],
                               c=labels_dbscan, cmap='Set1')
        axes[idx + 1].set_title(f'DBSCAN ε={eps}: {n_clusters} clusters, {n_noise} noise')
        axes[idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dbscan_comparison.png', dpi=150)
    plt.show()
    print("Saved: dbscan_comparison.png")

    print("\nKey Differences:")
    print("- K-Means: Assumes spherical clusters, must specify K")
    print("- DBSCAN: Finds arbitrary-shaped clusters, auto-detects eps")

# ==============================================================================
# EXAMPLE 4: FULL PIPELINE WITH GRID SEARCH
# ==============================================================================

def full_pipeline_gridsearch():
    """Complete pipeline with GridSearchCV for hyperparameter tuning."""

    print("\n" + "=" * 70)
    print("  EXAMPLE 4: FULL PIPELINE WITH HYPERPARAMETER TUNING")
    print("=" * 70)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline: Scale → Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ])

    # Define parameter grid
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto', 0.1, 1]
    }

    print("\nHyperparameter Search Space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,  # Use all cores
        verbose=1
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)

    print(f"Test Score: {test_score:.4f}")

    # Results visualization
    results_df = grid_search.cv_results_

    # Top 10 configurations
    print("\n" + "-" * 50)
    print("Top 10 Configurations:")
    print("-" * 50)

    indices = np.argsort(results_df['rank_test_score'])[:10]
    for rank, idx in enumerate(indices):
        params = results_df['params'][idx]
        mean_score = results_df['mean_test_score'][idx]
        std_score = results_df['std_test_score'][idx]

        print(f"\n{rank + 1}. Score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   Params: {params}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 2.5: SCIKIT-LEARN IN PRACTICE")
    print("=" * 70)

    # Example 1: Classification
    iris_results = iris_classification()

    # Example 2: Regression
    reg_results = diabetes_regression()

    # Example 3: Clustering
    clustering_example()

    # Example 4: Full pipeline with grid search
    full_pipeline_gridsearch()

    print("\n" + "=" * 70)
    print("  EXERCISES FOR THE READER")
    print("=" * 70)
    print("""
    1. CLASSIFICATION:
       - Try different datasets (breast_cancer, wine)
       - Implement your own k-fold cross-validation
       - Compare class weights for imbalanced data

    2. REGRESSION:
       - Try polynomial features + Ridge
       - Compare LASSO vs Ridge for feature selection
       - Build a pipeline with multiple preprocessing steps

    3. CLUSTERING:
       - Apply to customer segmentation data
       - Compare hierarchical clustering
       - Implement elbow method for optimal K

    4. GRID SEARCH:
       - Add feature selection to pipeline
       - Use RandomizedSearchCV for larger spaces
       - Implement your own scorer for specific metrics
    """)

    print("\n" + "=" * 70)
    print("  NEXT: Deep Learning (Neural Networks from Scratch)")
    print("=" * 70)

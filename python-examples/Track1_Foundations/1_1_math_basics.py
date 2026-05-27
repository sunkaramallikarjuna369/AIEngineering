"""
Track 1.1: Mathematics for AI
=============================
This file covers the essential mathematical foundations for AI/ML:
- Linear Algebra: vectors, matrices, dot products, eigenvalues
- Calculus: derivatives, chain rule, gradients
- Probability & Statistics: distributions, Bayes theorem, entropy

Author: AI Engineering Masterclass
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==============================================================================
# SECTION 1: LINEAR ALGEBRA
# ==============================================================================

def vectors_and_matrices():
    """
    Vectors and matrices are the backbone of neural networks.
    A vector is a 1D array, a matrix is a 2D array.

    In neural networks:
    - Input data → vector
    - Weights → matrix
    - Layer transformations → matrix-vector multiplication
    """
    # Create vectors
    v1 = np.array([1, 2, 3])  # Column vector (as row for display)
    v2 = np.array([4, 5, 6])

    # Create a matrix (2D array)
    # In ML, weights are stored as matrices
    W = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print("Vector 1:", v1)
    print("Vector 2:", v2)
    print("\nMatrix W (3x3):")
    print(W)

    # Dot product: measures similarity between vectors
    # Also the core operation in neural networks
    dot_product = np.dot(v1, v2)
    print(f"\nDot product v1 · v2 = {dot_product}")
    # Result: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    # Matrix-vector multiplication
    result = np.dot(W, v1)
    print(f"\nMatrix-vector multiplication W · v1 = {result}")

    # Matrix multiplication
    W_squared = np.dot(W, W)
    print(f"\nMatrix-matrix multiplication W · W:")
    print(W_squared)

    return dot_product, result

def matrix_operations():
    """Essential matrix operations in ML."""

    # Identity matrix (1s on diagonal)
    I = np.eye(3)
    print("Identity Matrix (3x3):")
    print(I)

    # Transpose (flip rows and columns)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    A_T = A.T
    print("\nMatrix A and its transpose:")
    print("A:\n", A)
    print("A.T:\n", A_T)

    # Determinant (used in eigenvalue calculation)
    B = np.array([[1, 2], [3, 4]])
    det_B = np.linalg.det(B)
    print(f"\nDeterminant of B: {det_B}")

    # Inverse (used in linear equations, pseudoinverse for SVD)
    B_inv = np.linalg.inv(B)
    print("Inverse of B:")
    print(B_inv)

    # Eigenvalues and eigenvectors
    # Critical for PCA, spectral clustering, and understanding matrix behavior
    eigenvalues, eigenvectors = np.linalg.eig(B)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # Singular Value Decomposition (SVD)
    # Used in: PCA, recommender systems, natural language processing
    U, S, Vt = np.linalg.svd(B)
    print(f"\nSVD Components:")
    print(f"U:\n{U}")
    print(f"S (singular values): {S}")
    print(f"Vt:\n{Vt}")

    return eigenvalues

def visualize_vectors():
    """Visualize vectors in 2D space."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Basic vectors
    ax1 = axes[0]
    ax1.quiver(0, 0, 1, 2, scale=1, color='blue', width=0.02)
    ax1.quiver(0, 0, 2, 1, scale=1, color='red', width=0.02)
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Vectors in 2D Space')
    ax1.legend(['v1 = [1,2]', 'v2 = [2,1]'])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dot product visualization
    ax2 = axes[1]
    v1 = np.array([3, 2])
    v2 = np.array([2, -1])

    # Project v1 onto v2
    v2_norm_sq = np.dot(v2, v2)
    projection = (np.dot(v1, v2) / v2_norm_sq) * v2

    ax2.quiver(0, 0, v1[0], v1[1], scale=1, color='blue', width=0.02)
    ax2.quiver(0, 0, v2[0], v2[1], scale=1, color='red', width=0.02)
    ax2.quiver(0, 0, projection[0], projection[1], scale=1, color='green',
               width=0.02, alpha=0.7)
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-2, 3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Dot Product = {np.dot(v1, v2)}')
    ax2.legend(['v1', 'v2', 'projection'])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Matrix transformation
    ax3 = axes[2]
    # Original unit square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    ax3.plot(square[0], square[1], 'b-', linewidth=2, label='Original')

    # Transformation matrix
    A = np.array([[2, 0.5], [0.5, 1.5]])
    transformed = A @ square
    ax3.plot(transformed[0], transformed[1], 'r-', linewidth=2,
             label='Transformed')
    ax3.set_xlim(-1, 4)
    ax3.set_ylim(-1, 3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Matrix Transformation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('linear_algebra_basics.png', dpi=150)
    plt.show()
    print("Saved: linear_algebra_basics.png")

# ==============================================================================
# SECTION 2: CALCULUS
# ==============================================================================

def derivatives():
    """
    Derivatives measure the rate of change.

    In ML:
    - Loss function derivative tells us how to adjust weights
    - Gradient descent uses derivatives to minimize loss
    """

    # Numerical derivative
    def f(x):
        return x**2  # f(x) = x²

    def numerical_derivative(f, x, h=1e-5):
        """Approximate derivative using finite differences."""
        return (f(x + h) - f(x - h)) / (2 * h)

    # Test at x=3: f'(x) = 2x, so f'(3) = 6
    x_val = 3.0
    exact = 2 * x_val  # Analytical: f'(x) = 2x
    approx = numerical_derivative(f, x_val)

    print(f"At x={x_val}:")
    print(f"  Exact derivative: {exact}")
    print(f"  Numerical approximation: {approx}")
    print(f"  Error: {abs(exact - approx):.2e}")

    return approx

def chain_rule():
    """
    Chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)

    This is the FOUNDATION of backpropagation!
    When computing loss gradient through layers:
    - Layer 3 gradient × Layer 2 gradient × Layer 1 gradient
    """

    # Example: f(g(x)) where g(x) = 2x + 1 and f(u) = u²
    # f(g(x)) = (2x + 1)²
    # f'(x) = 2 * (2x + 1) * 2 = 8x + 4

    def composite(x):
        g = 2*x + 1  # Inner function
        return g**2   # Outer function

    def chain_rule_derivative(x):
        """Using chain rule: 2*(2x+1)*2"""
        return 8*x + 4

    x_val = 2.0
    result = composite(x_val)
    grad = chain_rule_derivative(x_val)

    print(f"Composite function f(g(x)) = (2x + 1)²")
    print(f"At x={x_val}: f(g(x)) = {result}")
    print(f"Derivative using chain rule: {grad}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: The function
    x = np.linspace(-2, 4, 100)
    y = (2*x + 1)**2
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.scatter([x_val], [result], color='red', s=100, zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(g(x))')
    ax1.set_title('Composite Function f(g(x)) = (2x+1)²')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient as slope
    x_close = np.linspace(x_val - 1, x_val + 1, 50)
    y_close = (2*x_close + 1)**2
    tangent_x = np.linspace(x_val - 0.5, x_val + 0.5, 20)
    tangent_y = result + grad * (tangent_x - x_val)

    ax2.plot(x_close, y_close, 'b-', linewidth=2)
    ax2.plot(tangent_x, tangent_y, 'r--', linewidth=2, label=f'Tangent (slope={grad})')
    ax2.scatter([x_val], [result], color='red', s=100, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(g(x))')
    ax2.set_title(f'Derivative = Slope = {grad}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('chain_rule.png', dpi=150)
    plt.show()
    print("Saved: chain_rule.png")

    return grad

def gradient_descent():
    """
    Gradient descent: repeatedly move in direction of steepest descent.

    Update rule: θ_new = θ_old - LearningRate × ∇L(θ)

    This is how neural networks LEARN!
    """

    def loss_surface(x, y):
        """A simple paraboloid loss surface: L = x² + y²"""
        return x**2 + y**2

    def gradient(x, y):
        """Gradient of paraboloid: ∂L/∂x = 2x, ∂L/∂y = 2y"""
        return np.array([2*x, 2*y])

    # Starting point
    x, y = 3.0, 4.0
    learning_rate = 0.1

    # Store trajectory for visualization
    trajectory = [(x, y)]
    losses = [loss_surface(x, y)]

    print(f"Starting point: ({x}, {y})")
    print(f"Starting loss: {losses[-1]}")
    print("\nGradient Descent Steps:")
    print("-" * 40)

    # Run gradient descent
    for step in range(50):
        grad = gradient(x, y)
        grad_norm = np.linalg.norm(grad)

        # Stop if gradient is small enough (converged)
        if grad_norm < 0.001:
            print(f"\nConverged at step {step}!")
            break

        # Update rule
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]

        trajectory.append((x, y))
        losses.append(loss_surface(x, y))

        if step < 10 or step % 10 == 0:
            print(f"Step {step:2d}: ({x:.4f}, {y:.4f}), loss={losses[-1]:.6f}")

    print(f"\nFinal point: ({x:.6f}, {y:.6f})")
    print(f"Final loss: {losses[-1]:.10f}")

    return trajectory, losses

def visualize_gradient_descent():
    """Visualize gradient descent on a loss surface."""

    # Create loss surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Run gradient descent
    trajectory, losses = gradient_descent()

    # Create figure
    fig = plt.figure(figsize=(14, 5))

    # Plot 1: 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Loss')

    trajectory_arr = np.array(trajectory)
    ax1.plot(trajectory_arr[:, 0], trajectory_arr[:, 1], 'r.-',
             linewidth=1, markersize=4, label='Gradient Descent')
    ax1.scatter([trajectory[0][0]], [trajectory[0][1]], color='red', s=100,
                marker='o', zorder=5, label='Start')
    ax1.scatter([trajectory[-1][0]], [trajectory[-1][1]], color='white', s=100,
                marker='*', zorder=5, label='End')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Descent on Loss Surface L = x² + y²')
    ax1.legend()

    # Plot 2: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax2.scatter(trajectory_arr[:, 0], trajectory_arr[:, 1], losses,
                color='red', s=50, marker='o', zorder=5)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Surface (3D)')

    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=150)
    plt.show()
    print("Saved: gradient_descent.png")

# ==============================================================================
# SECTION 3: PROBABILITY & STATISTICS
# ==============================================================================

def probability_basics():
    """
    Probability is fundamental to ML:
    - Neural networks output probability distributions
    - Cross-entropy loss uses probabilities
    - Bayesian methods for uncertainty quantification
    """

    # Discrete distributions
    # Uniform distribution: all outcomes equally likely
    uniform = np.ones(6) / 6  # Fair die
    print("Uniform Distribution (fair die):")
    print(f"  P(x) = 1/6 for x in {{1,2,3,4,5,6}}")
    print(f"  Sum of probabilities: {np.sum(uniform)}")

    # Binomial distribution: number of successes in n trials
    from scipy import stats

    n, p = 10, 0.5  # 10 trials, 50% success probability
    binomial = stats.binom(n, p)

    print(f"\nBinomial Distribution (n={n}, p={p}):")
    print(f"  Expected value: {binomial.mean()}")
    print(f"  Variance: {binomial.var()}")
    print(f"  P(X=5): {binomial.pmf(5):.4f}")

    # Normal (Gaussian) distribution
    mu, sigma = 0, 1
    normal = stats.norm(mu, sigma)

    print(f"\nNormal Distribution (μ={mu}, σ={sigma}):")
    print(f"  PDF at x=0: {normal.pdf(0):.4f}")
    print(f"  CDF at F(0): {normal.cdf(0):.4f}")

    return uniform, binomial, normal

def bayes_theorem():
    """
    Bayes Theorem: P(A|B) = P(B|A) × P(A) / P(B)

    This is the FOUNDATION of:
    - Naive Bayes classifiers
    - Bayesian neural networks
    - Uncertainty quantification
    - Medical diagnosis (screening tests)
    """

    print("Bayes Theorem Demonstration")
    print("=" * 50)
    print("Scenario: Medical test for a rare disease")
    print("-" * 50)

    # Prior probability (prevalence)
    P_disease = 0.01  # 1% of population has the disease

    # Test accuracy
    P_positive_given_disease = 0.99  # 99% sensitive
    P_negative_given_healthy = 0.95  # 95% specific

    # Calculate using Bayes theorem
    # P(disease|+) = P(+|disease) × P(disease) / P(+)

    # P(+) = P(+|disease) × P(disease) + P(+|healthy) × P(healthy)
    P_positive_given_healthy = 1 - P_negative_given_healthy
    P_healthy = 1 - P_disease

    P_positive = (P_positive_given_disease * P_disease +
                  P_positive_given_healthy * P_healthy)

    # Apply Bayes theorem
    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

    print(f"P(Disease) = {P_disease} (prior)")
    print(f"P(+|Disease) = {P_positive_given_disease} (sensitivity)")
    print(f"P(-|Healthy) = {P_negative_given_healthy} (specificity)")
    print()
    print(f"P(+) = {P_positive:.4f}")
    print(f"P(Disease|+) = {P_disease_given_positive:.4f}")
    print()
    print("Interpretation:")
    print(f"  If you test positive, you have {P_disease_given_positive*100:.1f}% chance")
    print(f"  of actually having the disease!")
    print()
    print("Why so low? Because most positive results come from")
    print("healthy people (since 99% are healthy).")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(-5, 5, 200)
    healthy_dist = stats.norm(-1, 1)
    disease_dist = stats.norm(1.5, 0.8)

    ax.fill_between(x, healthy_dist.pdf(x) * 0.99, alpha=0.3,
                    label='Healthy (99%)', color='green')
    ax.fill_between(x, disease_dist.pdf(x) * 0.01 + healthy_dist.pdf(x) * 0.99,
                    alpha=0.3, label='Disease (1%)', color='red')
    ax.plot(x, healthy_dist.pdf(x) * 0.99, 'g-', linewidth=2)
    ax.plot(x, disease_dist.pdf(x) * 0.01 + healthy_dist.pdf(x) * 0.99, 'r-', linewidth=2)

    ax.axvline(x=0, color='black', linestyle='--', label='Decision threshold')
    ax.fill_between(x, disease_dist.pdf(x) * 0.01 + healthy_dist.pdf(x) * 0.99,
                    healthy_dist.pdf(x) * 0.99, where=(x > 0),
                    alpha=0.2, color='purple', label='False positives')

    ax.set_xlabel('Test Score')
    ax.set_ylabel('Probability Density')
    ax.set_title('Bayes Theorem: Test Result Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayes_theorem.png', dpi=150)
    plt.show()
    print("Saved: bayes_theorem.png")

    return P_disease_given_positive

def entropy():
    """
    Entropy: measures uncertainty/information content.
    H(X) = -Σ P(x) × log₂(P(x))

    Used in:
    - Decision trees (ID3, C4.5) - find splits that minimize entropy
    - Information gain = entropy before - entropy after
    - Cross-entropy loss in neural networks
    """

    from scipy.stats import entropy

    print("Entropy Demonstration")
    print("=" * 50)

    # Binary distribution with varying probability
    p = np.linspace(0.01, 0.99, 50)
    entropies = [entropy([p_i, 1-p_i], base=2) for p_i in p]

    print("Binary Entropy H(p) for p ∈ [0.01, 0.99]:")
    print(f"  Maximum entropy when p=0.5: H = {max(entropies):.4f} bits")
    print(f"  Minimum entropy when p≈0 or p≈1: H ≈ 0 bits")

    # Example: spam classification
    # Before split: 50% spam, 50% ham
    before = [0.5, 0.5]
    H_before = entropy(before, base=2)

    # After split by word "FREE"
    # Emails with "FREE": 80% spam, 20% ham
    with_free = [0.8, 0.2]
    H_with = entropy(with_free, base=2)

    # Emails without "FREE": 20% spam, 80% ham
    without_free = [0.2, 0.8]
    H_without = entropy(without_free, base=2)

    # Weighted average after split
    P_with_free = 0.3  # 30% of emails have "FREE"
    H_after = P_with_free * H_with + (1 - P_with_free) * H_without

    information_gain = H_before - H_after

    print(f"\nSpam Detection Example:")
    print(f"  Before split: H = {H_before:.4f} bits")
    print(f"  After split:  H = {H_after:.4f} bits")
    print(f"  Information Gain = {information_gain:.4f} bits")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Binary entropy curve
    ax1.plot(p, entropies, 'b-', linewidth=2)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Max entropy = 1 bit')
    ax1.set_xlabel('P(class 1)')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Binary Entropy Function H(p)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Information gain tree
    ax2.axis('off')

    # Root node
    ax2.text(0.5, 0.9, f'Root Node\nH = {H_before:.2f} bits',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Split nodes
    ax2.annotate('', xy=(0.25, 0.6), xytext=(0.45, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax2.annotate('', xy=(0.75, 0.6), xytext=(0.55, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))

    ax2.text(0.25, 0.55, f'"FREE" found\nH = {H_with:.2f} bits',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax2.text(0.75, 0.55, f'"FREE" absent\nH = {H_without:.2f} bits',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax2.text(0.5, 0.25, f'Information Gain = {information_gain:.4f} bits',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange'))

    ax2.set_title('Information Gain from Split')

    plt.tight_layout()
    plt.savefig('entropy.png', dpi=150)
    plt.show()
    print("Saved: entropy.png")

    return entropies

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 1.1: MATHEMATICS FOR AI - COMPLETE OVERVIEW")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  SECTION 1: LINEAR ALGEBRA")
    print("=" * 70)
    dot, result = vectors_and_matrices()
    visualize_vectors()

    print("\n" + "=" * 70)
    print("  SECTION 2: CALCULUS")
    print("=" * 70)
    deriv = derivatives()
    grad = chain_rule()
    visualize_gradient_descent()

    print("\n" + "=" * 70)
    print("  SECTION 3: PROBABILITY & STATISTICS")
    print("=" * 70)
    uniform, binomial, normal = probability_basics()
    bayes_prob = bayes_theorem()
    entropies = entropy()

    print("\n" + "=" * 70)
    print("  EXERCISES FOR THE READER")
    print("=" * 70)
    print("""
    1. LINEAR ALGEBRA:
       - Create a 4x4 matrix and compute its eigenvalues
       - Implement matrix multiplication from scratch
       - Visualize how different transformation matrices affect a square

    2. CALCULUS:
       - Implement gradient descent for L = x^4 - 2x^2 + 1
       - Compare different learning rates (0.01, 0.1, 0.5)
       - Visualize the loss surface for a 2D Gaussian

    3. PROBABILITY:
       - Simulate 1000 coin tosses, calculate observed probabilities
       - Implement Naive Bayes classifier for spam detection
       - Calculate mutual information between two variables
    """)

    print("\n" + "=" * 70)
    print("  NEXT: Python for AI (NumPy, Pandas, Matplotlib)")
    print("=" * 70)

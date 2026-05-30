"""
Track 2.1: Machine Learning Basics - Beginners Edition
=====================================================
What is ML? Types of ML? Simple examples to understand.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT IS MACHINE LEARNING?")
print("=" * 60)

print("""
[?] TRADITIONAL PROGRAMMING:
   You write rules -> Computer follows rules -> Gets answer

   Example: Detect spam
   IF "FREE" in email AND "CLICK" in email THEN spam
   IF "meeting" in email AND "tomorrow" in email THEN not spam

[BOT] MACHINE LEARNING:
   You give data + answers -> Computer learns rules -> Predicts new answers

   Example: Detect spam
   Show 10,000 spam emails -> Model learns patterns
   Show new email -> Model predicts "spam" or "not spam"
""")

# ==============================================================================
# PART 1: THE ML RECIPE
# ==============================================================================

print("\n" + "=" * 60)
print("  THE ML RECIPE (How ML Works)")
print("=" * 60)

print("""
   [LIST] ML Has 4 Ingredients:

   1. DATA
      - Features (inputs): what you know
      - Labels (outputs): what you want to predict

   2. MODEL
      - A function that transforms inputs to outputs
      - Like a math formula: y = mx + b

   3. LOSS FUNCTION
      - Measures how wrong predictions are
      - Goal: Minimize this!

   4. OPTIMIZER
      - Adjusts model to reduce loss
      - Like a teacher grading and guiding improvement

   [LOOP] THE LOOP:
   Data -> Model -> Predictions -> Loss -> Update Model -> Repeat 1000x
""")

# ==============================================================================
# PART 2: TYPES OF ML
# ==============================================================================

print("\n" + "=" * 60)
print("  TYPES OF MACHINE LEARNING")
print("=" * 60)

print("""
   1. SUPERVISED LEARNING (Learn with examples)
      |- Classification: Predict categories (spam/not spam)
      |- Regression: Predict numbers (house price)

   2. UNSUPERVISED LEARNING (Find patterns)
      |- Clustering: Group similar items
      |- Dimensionality Reduction: Simplify data

   3. REINFORCEMENT LEARNING (Learn from experience)
      |- Agent learns by trial and reward
""")

# ==============================================================================
# PART 3: SUPERVISED LEARNING EXAMPLES
# ==============================================================================

print("\n" + "=" * 60)
print("  SUPERVISED LEARNING - Classification")
print("=" * 60)

from sklearn.datasets import make_blobs
import numpy as np

# Create simple data: 2 groups of points
print("\n1. Creating Data:")
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
print(f"   Data points: {len(X)}")
print(f"   Features per point: {X.shape[1]}")
print(f"   Categories: {np.unique(y)}")
print(f"   (0 = red group, 1 = blue group)")

# Look at first few points
print(f"\n   First 5 data points:")
for i in range(5):
    print(f"   Point {i+1}: X={X[i]}, Label={y[i]}")

print("\n2. Training a Classifier:")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"   Training samples: {len(X_train)}")
print(f"   Test accuracy: {accuracy:.1%}")

print("\n3. Making New Predictions:")
new_point = np.array([[0.5, 0.5]])
prediction = model.predict(new_point)
print(f"   New point: {new_point}")
print(f"   Prediction: Class {prediction[0]}")

# ==============================================================================
# PART 4: REGRESSION EXAMPLE
# ==============================================================================

print("\n" + "=" * 60)
print("  SUPERVISED LEARNING - Regression")
print("=" * 60)

print("\n1. Regression: Predict Numbers")
print("   Example: Predict house price from size")

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simple data: house sizes → prices
house_sizes = np.array([500, 750, 1000, 1250, 1500, 1750, 2000]).reshape(-1, 1)
house_prices = np.array([100, 150, 200, 250, 300, 350, 400])

# Train model
model = LinearRegression()
model.fit(house_sizes, house_prices)

print(f"   Model learned: Price = {model.coef_[0]:.2f} × Size + {model.intercept_:.2f}")

# Predict
test_sizes = np.array([800, 1200, 1800]).reshape(-1, 1)
predicted = model.predict(test_sizes)

print("\n   Predictions:")
for size, price in zip(test_sizes, predicted):
    print(f"   {size[0]} sq ft → ${price[0]:.0f}K")

# Quick visualization (uncomment to see)
# plt.scatter(house_sizes, house_prices, color='blue', label='Data')
# plt.plot(test_sizes, predicted, color='red', label='Predictions')
# plt.xlabel('Size (sq ft)')
# plt.ylabel('Price ($K)')
# plt.legend()
# plt.show()

# ==============================================================================
# PART 5: UNSUPERVISED LEARNING - CLUSTERING
# ==============================================================================

print("\n" + "=" * 60)
print("  UNSUPERVISED LEARNING - Clustering")
print("=" * 60)

print("\n1. Clustering: Group Similar Items")
print("   Example: Group customers by behavior")

from sklearn.cluster import KMeans
import numpy as np

# Customer data: [age, spending_score]
customers = np.array([
    [20, 80], [22, 90], [25, 85],  # Young, high spending
    [45, 30], [50, 40], [55, 35],  # Older, low spending
    [30, 50], [35, 55], [40, 45],  # Middle, medium spending
])

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
groups = kmeans.fit_predict(customers)

print(f"   Customers grouped into {len(np.unique(groups))} clusters")
print(f"   Cluster centers:\n{kmeans.cluster_centers_}")

print("\n   Customer assignments:")
for i, (customer, group) in enumerate(zip(customers, groups)):
    print(f"   Customer {i+1}: Age={customer[0]}, Spending={customer[1]} → Group {group}")

# ==============================================================================
# WHAT'S NEXT
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
[OK] ML learns patterns from data, not rules from programmers

[OK] Supervised = Learning with labeled examples
   - Classification: predict category
   - Regression: predict number

[OK] Unsupervised = Finding patterns in unlabeled data
   - Clustering: group similar items

[OK] The ML Loop: Data -> Model -> Loss -> Update -> Repeat

NEXT: Learn specific algorithms (Linear Regression, Decision Trees, etc.)
""")
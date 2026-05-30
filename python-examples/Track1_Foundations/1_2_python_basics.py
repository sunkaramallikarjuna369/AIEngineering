"""
Track 1.2: Python for AI - Beginners Edition
============================================
Simple, clear examples of Python libraries used in AI/ML.

Author: AI Engineering Masterclass
"""

# ==============================================================================
# PART 1: NUMPY - THE BASICS
# ==============================================================================

print("=" * 60)
print("  NUMPY - Numerical Python")
print("=" * 60)

# NumPy is like a calculator but for arrays of numbers
# It's MUCH faster than regular Python lists

import numpy as np

# 1. Creating arrays (like lists, but better!)
print("\n[1] Creating Arrays:")
numbers = np.array([1, 2, 3, 4, 5])
print(f"   Array: {numbers}")
print(f"   Type: {type(numbers)}")

# 2. Math operations (element-wise!)
print("\n[2] Math Operations:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"   {a} + {b} = {a + b}")  # Addition
print(f"   {a} * 2 = {a * 2}")   # Multiplication
print(f"   {a} ** 2 = {a ** 2}") # Squaring

# 3. Statistics (easy!)
print("\n[3] Statistics:")
data = np.array([10, 20, 30, 40, 50])
print(f"   Data: {data}")
print(f"   Sum: {data.sum()}")
print(f"   Mean: {data.mean()}")
print(f"   Max: {data.max()}")
print(f"   Min: {data.min()}")

# 4. 2D arrays (matrices!)
print("\n[4] Matrices:")
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"   Shape: {matrix.shape}  # 2 rows, 3 columns")
print(f"   Matrix:\n{matrix}")

# 5. Matrix multiplication
print("\n[5] Matrix Multiplication:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.dot(A, B)
print(f"   A @ B =\n{result}")

# ==============================================================================
# PART 2: PANDAS - DATA HANDLING
# ==============================================================================

print("\n" + "=" * 60)
print("  PANDAS - Data Handling")
print("=" * 60)

# Pandas is like Excel but in Python
# Great for tables of data!

import pandas as pd

# 1. Creating a DataFrame (like a table)
print("\n[1] Creating a Table:")
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 92, 78]
}
df = pd.DataFrame(data)
print(df)

# 2. Viewing data
print("\n[2] Quick View:")
print(f"   First 2 rows:\n{df.head(2)}")
print(f"   Columns: {df.columns.tolist()}")

# 3. Selecting columns
print("\n[3] Selecting Columns:")
print(f"   Names: {df['Name'].tolist()}")
print(f"   Ages: {df['Age'].tolist()}")

# 4. Filtering rows
print("\n[4] Filtering (like Excel IF):")
high_scorers = df[df['Score'] > 80]
print(f"   People with Score > 80:")
print(high_scorers)

# 5. Simple math on columns
print("\n[5] Column Math:")
df['Bonus'] = df['Score'] * 0.1  # Add 10% bonus
df['Pass'] = df['Score'] > 80    # Pass/Fail
print(df)

# ==============================================================================
# PART 3: MATPLOTLIB - VISUALIZATION
# ==============================================================================

print("\n" + "=" * 60)
print("  MATPLOTLIB - Drawing Charts")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt

# 1. Line chart
print("\n[1] Line Chart:")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-o', label='y = 2x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Line Chart')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('line_chart.png')  # Save instead of show
print("   Saved as line_chart.png")

# 2. Scatter plot
print("\n[2] Scatter Plot:")
x = np.random.randn(50)
y = np.random.randn(50)
plt.figure(figsize=(8, 4))
plt.scatter(x, y, alpha=0.5, c='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Scatter')
plt.savefig('scatter_plot.png')
print("   Saved as scatter_plot.png")

# 3. Bar chart
print("\n[3] Bar Chart:")
categories = ['A', 'B', 'C', 'D']
values = [30, 45, 25, 60]
plt.figure(figsize=(8, 4))
plt.bar(categories, values, color=['red', 'blue', 'green', 'purple'])
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Category Comparison')
plt.savefig('bar_chart.png')
print("   Saved as bar_chart.png")
plt.close('all')

print("\n   Charts saved! Open them to see the visualizations.")

# ==============================================================================
# PART 4: SIMPLE ML WITH SCIKIT-LEARN
# ==============================================================================

print("\n" + "=" * 60)
print("  SCIKIT-LEARN - Your First ML Model")
print("=" * 60)

from sklearn.linear_model import LinearRegression

# We want to predict house prices based on size
# Imagine: 5 houses with their sizes (sq ft)

print("\n[1] The Problem:")
X = np.array([[500], [750], [1000], [1250], [1500]])  # Size in sq ft
y = np.array([100, 150, 200, 250, 300])  # Prices in $1000s

print(f"   House sizes: {X.flatten()}")
print(f"   Prices: {y}")

print("\n[2] Creating & Training Model:")
model = LinearRegression()  # The model
model.fit(X, y)  # "Learning" from data

print(f"   Learned slope: {model.coef_[0]:.2f}")
print(f"   Learned intercept: {model.intercept_:.2f}")
print(f"   Formula: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")

print("\n[3] Making Predictions:")
new_sizes = np.array([[800], [1100], [2000]])
predictions = model.predict(new_sizes)
print(f"   For 800 sq ft: ${predictions[0]*1000:.0f}")
print(f"   For 1100 sq ft: ${predictions[1]*1000:.0f}")
print(f"   For 2000 sq ft: ${predictions[2]*1000:.0f}")

# ==============================================================================
# PART 5: PYTHON TIPS FOR AI
# ==============================================================================

print("\n" + "=" * 60)
print("  PYTHON TIPS FOR AI")
print("=" * 60)

# 1. List comprehension (concise loops)
print("\n[1] List Comprehension:")
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"   {numbers} -> squared = {squared}")

# 2. Dict comprehension
print("\n[2] Dictionary Comprehension:")
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]
score_dict = {name: score for name, score in zip(names, scores)}
print(f"   {score_dict}")

# 3. Lambda functions (short functions)
print("\n[3] Lambda Functions:")
square = lambda x: x ** 2
print(f"   square(5) = {square(5)}")

# 4. Map (apply function to all items)
print("\n[4] Map Function:")
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(f"   {numbers} -> doubled = {doubled}")

# 5. Filter (keep items matching condition)
print("\n[5] Filter Function:")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"   Numbers 1-10: {numbers}")
print(f"   Even numbers: {evens}")

# ==============================================================================
# WHAT'S NEXT
# ==============================================================================

print("\n" + "=" * 60)
print("  NEXT STEPS")
print("=" * 60)
print("""
Now you know the basics! Next track will cover:
    [+] Linear Algebra (vectors, matrices)
    [+] Calculus (derivatives, gradients)
    [+] Probability (statistics)

Run this file with:
    python python-examples/Track1_Foundations/1_2_python_basics.py
""")
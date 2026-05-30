"""
Track 1.3: Data Engineering - Beginners Edition
================================================
How to prepare data for ML models.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT IS DATA ENGINEERING?")
print("=" * 60)

print("""
[DATA] DATA ENGINEERING = Preparing data for ML

   Think of it like cooking:

   [COOK] RAW INGREDIENTS = Raw data (messy, unstructured)
   [NOTE] RECIPE = Data processing pipeline
   [OK] CLEAN DISH = Ready-to-use data for ML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   WHY IT MATTERS:

   "Garbage in, garbage out"

   If data is bad → Model is bad

   Real stats:
   - Data scientists spend 80% of time on data prep
   - Only 20% on actual modeling!
""")

# ==============================================================================
# PART 1: DATA TYPES
# ==============================================================================

print("\n" + "=" * 60)
print("  TYPES OF DATA")
print("=" * 60)

print("""
   [FOLDER] STRUCTURED DATA (Tables)
   |- Like Excel/CSV files
   |- Rows and columns
   |- Easy to process
   Example: Customer database

   [FILE] SEMI-STRUCTURED (Flexible format)
   |- JSON, XML
   |- Nested information
   |- API responses
   Example: API responses

   [IMG] UNSTRUCTURED DATA (No fixed format)
   |- Images, audio, video
   |- Text documents
   |- Harder to process
   Example: PDFs, social media posts
""")

# ==============================================================================
# PART 2: DATA LOADING
# ==============================================================================

import pandas as pd

def data_loading():
    """Load and explore data."""
    print("\n[RUN] LOADING DATA")
    print("-" * 40)

    # Create sample data
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 40],
        'salary': [50000, 60000, 75000, 55000, 80000],
        'department': ['Sales', 'IT', 'HR', 'Sales', 'IT']
    }
    df = pd.DataFrame(data)

    print("Sample employee data:")
    print(df)
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Basic exploration
    print("\n[DATA] Data Info:")
    print(df.info())

    print("\n[CHART] Statistics:")
    print(df.describe())

data_loading()

# ==============================================================================
# PART 3: DATA CLEANING
# ==============================================================================

def data_cleaning():
    """Clean messy data."""
    print("\n[CLEAN] DATA CLEANING")
    print("-" * 40)

    # Create messy data
    data = {
        'name': ['Alice', 'Bob', None, 'Diana', '  Eve  '],
        'age': [25, 35, -5, 28, 150],  # -5 and 150 are outliers!
        'email': ['alice@work.com', 'bob@work', 'charlie@work.com', None, 'eve@work.com']
    }
    df = pd.DataFrame(data)

    print("Messy data:")
    print(df)

    print("\n[SETUP] Cleaning steps:")

    # 1. Handle missing values
    print("1. Missing values:")
    print(f"   Before: {df.isnull().sum().sum()} missing")
    df['name'].fillna('Unknown', inplace=True)
    df['age'].fillna(df['age'].median(), inplace=True)
    df['email'].fillna('no_email@unknown.com', inplace=True)
    print(f"   After: {df.isnull().sum().sum()} missing")

    # 2. Fix outliers
    print("\n2. Fix outliers:")
    df['age'] = df['age'].clip(18, 100)  # Age between 18 and 100
    print(f"   Ages fixed: [25, 35, 35 (was -5), 28, 40 (was 150)]")

    # 3. Clean strings
    print("\n3. Clean strings:")
    df['name'] = df['name'].str.strip()  # Remove whitespace
    print(f"   Name trimmed: '  Eve  ' → 'Eve'")

    print("\nClean data:")
    print(df)

data_cleaning()

# ==============================================================================
# PART 4: FEATURE ENGINEERING
# ==============================================================================

def feature_engineering():
    """Create useful features from raw data."""
    print("\n[GEAR] FEATURE ENGINEERING")
    print("-" * 40)

    data = {
        'purchase_date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'],
        'amount': [100, 250, 75, 500],
        'customer_id': [1, 2, 1, 3]
    }
    df = pd.DataFrame(data)

    print("Original data:")
    print(df)

    # 1. Date features
    print("\n1. Date features:")
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['day_of_week'] = df['purchase_date'].dt.day_name()
    print(f"   Added: month, day_of_week")

    # 2. Amount categories
    print("\n2. Amount categories:")
    df['amount_category'] = pd.cut(df['amount'],
                                   bins=[0, 100, 300, float('inf')],
                                   labels=['Low', 'Medium', 'High'])
    print(df[['amount', 'amount_category']])

    # 3. Aggregation
    print("\n3. Aggregation (customer stats):")
    customer_stats = df.groupby('customer_id')['amount'].agg(['sum', 'mean', 'count'])
    customer_stats.columns = ['total_spent', 'avg_purchase', 'num_purchases']
    print(customer_stats)

# ==============================================================================
# PART 5: TRAIN/TEST SPLIT
# ==============================================================================

def train_test_split_demo():
    """Split data for training and testing."""
    print("\n[SPLIT] TRAIN/TEST SPLIT")
    print("-" * 40)

    # Create data
    X = list(range(100))  # Features
    y = [x * 2 + random.randint(-5, 5) for x in X]  # Target

    # Split
    import random
    random.seed(42)
    y = [x * 2 + random.randint(-5, 5) for x in X]

    # 80/20 split
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Total data: {len(X)} samples")
    print(f"Training: {len(X_train)} samples (80%)")
    print(f"Testing: {len(X_test)} samples (20%)")

    print("\n[TARGET] Why split?")
    print("   - Training: Learn patterns")
    print("   - Testing: Check if model generalizes")
    print("   - Never test on training data!")

# ==============================================================================
# PART 6: NORMALIZATION
# ==============================================================================

def normalization():
    """Scale features to similar ranges."""
    print("\n[SCALE] NORMALIZATION / SCALING")
    print("-" * 40)

    import numpy as np

    # Create data with different scales
    data = {
        'height_cm': [160, 170, 180, 165, 175],
        'weight_kg': [60, 80, 75, 65, 70],
        'age': [25, 45, 35, 28, 55]
    }
    df = pd.DataFrame(data)

    print("Data with different scales:")
    print(df)
    print(f"\nHeight: {df['height_cm'].min()}-{df['height_cm'].max()}")
    print(f"Weight: {df['weight_kg'].min()}-{df['weight_kg'].max()}")
    print(f"Age: {df['age'].min()}-{df['age'].max()}")

    print("\n[SETUP] Min-Max Scaling (0 to 1):")
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f'{col}_scaled'] = (df[col] - min_val) / (max_val - min_val)

    print(df[['height_cm_scaled', 'weight_kg_scaled', 'age_scaled']])

# ==============================================================================
# RUN ALL DEMOS
# ==============================================================================

print("\n" + "=" * 60)
print("  RUNNING ALL DEMOS")
print("=" * 60)

data_loading()
data_cleaning()
feature_engineering()
train_test_split_demo()
normalization()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
[OK] Data Engineering = 80% of ML work!

[OK] Data types: Structured, Semi-structured, Unstructured

[OK] Cleaning: Fix missing, outliers, strings

[OK] Feature Engineering: Create useful features from raw data

[OK] Always split: Training vs Testing (80/20 typical)

[OK] Good data > Good algorithms

NEXT: Start building ML models!
""")
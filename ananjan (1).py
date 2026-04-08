# =============================================================================
#  Banking Fraud Detection – VUIP111 Major Project
#  File     : ananjan.py
#  Author   : Ananjan
#  Section  : Imports & Setup | Load Dataset | Handle Class Imbalance | Train/Test Split
# =============================================================================


# =============================================================================
#  IMPORTS & SETUP
# =============================================================================
#
#  Every library used across the full pipeline is imported here so there are
#  no hidden dependencies later in the code.
#
#  pandas / numpy        → data loading, manipulation, and array operations
#  matplotlib / seaborn  → all visualisations (charts, heatmaps, etc.)
#  sklearn               → ML algorithms, preprocessing, evaluation metrics
#  sklearn.utils.resample→ manual oversampling to fix class imbalance
#  joblib                → serialising and saving the trained model to disk
#  json / os             → saving metadata and managing file paths
# =============================================================================

import warnings
warnings.filterwarnings('ignore')   # keep the output clean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils import resample

import joblib
import json
import os

# Global plot style – applied once here so every chart looks consistent
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)

print('All libraries imported successfully.')
print(f'   pandas {pd.__version__}  |  numpy {np.__version__}')


# =============================================================================
#  LOAD DATASET
# =============================================================================
#
#  We read the raw CSV into a pandas DataFrame.
#  After loading we immediately verify:
#    • The shape (rows × columns) to confirm nothing was lost on read
#    • The column names so we know what we are working with
#    • The first few rows to visually sanity-check the data
#
#  UPDATE the DATA_PATH variable if your CSV is stored in a different folder.
# =============================================================================

DATA_PATH = 'banking_fraud_dataset_50k.csv'

df = pd.read_csv(DATA_PATH)

print(f'\nDataset loaded  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns')
print('\nColumn names:')
print(df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())

# Quick sanity check on data types – helps catch encoding issues early
print('\nData types:')
print(df.dtypes)

# Check for any nulls before we do anything else
total_nulls = df.isnull().sum().sum()
if total_nulls == 0:
    print('\nNo missing values detected in the raw dataset.')
else:
    print(f'\nWarning: {total_nulls} missing values found. Will be handled during preprocessing.')


# =============================================================================
#  HANDLE CLASS IMBALANCE & TRAIN / TEST SPLIT
# =============================================================================
#
#  Why class imbalance matters:
#  ─────────────────────────────
#  Only ~3.5% of the 50,000 transactions are fraudulent.
#  If we train a model on this raw split, it can achieve ~96.5% accuracy
#  by simply predicting "Legitimate" every single time — a completely
#  useless model for a fraud detection system.
#
#  Our approach:
#  ─────────────
#  Step 1 → Split the data FIRST (80% train, 20% test), using stratify=y
#            so both splits have the same fraud ratio (~3.5%).
#
#  Step 2 → Apply oversampling ONLY to the training set.
#            We use sklearn's resample() to duplicate (with replacement)
#            minority-class (fraud) rows until the training set is balanced.
#
#  Step 3 → Leave the TEST set completely untouched.
#            The test set must reflect real-world distribution so our
#            evaluation metrics are honest.
#
#  This order is critical. Oversampling before splitting is a form of
#  data leakage — synthetic examples from the training set bleed into
#  the test set and inflate performance numbers dishonestly.
# =============================================================================

# ── Feature / target separation ───────────────────────────────────────────────
#
#  Before splitting we need the cleaned & engineered DataFrame (df_clean).
#  This file assumes df_clean is prepared by alwin.py and siddharth.py.
#  For standalone execution, run those files first.
#
#  Uncomment the line below if running this file on its own after loading
#  df_clean from disk:
#
#  df_clean = pd.read_csv('df_clean_engineered.csv')

X = df_clean.drop(columns=['is_fraud'])
y = df_clean['is_fraud']

print(f'\nFeatures shape : {X.shape}')
print(f'Target shape   : {y.shape}')
print(f'\nClass distribution before balancing:')
print(y.value_counts())
print(f'Fraud percentage: {y.mean() * 100:.2f}%')

# ── Train / Test Split (80 / 20, stratified) ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # ensures both splits keep the ~3.5% fraud ratio
)

print(f'\nTrain size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}')
print(f'Fraud in train : {y_train.sum():,}  ({y_train.mean() * 100:.2f}%)')
print(f'Fraud in test  : {y_test.sum():,}   ({y_test.mean() * 100:.2f}%)')

# ── Oversampling – minority class in training set only ────────────────────────
#
#  We combine X_train and y_train into a single DataFrame so we can
#  separate by class label, then resample the minority (fraud) class.

train_data = pd.concat([X_train, y_train], axis=1)

majority = train_data[train_data['is_fraud'] == 0]    # legitimate transactions
minority = train_data[train_data['is_fraud'] == 1]    # fraud transactions

print(f'\nBefore oversampling:')
print(f'  Majority (Legit) : {len(majority):,}')
print(f'  Minority (Fraud) : {len(minority):,}')

minority_upsampled = resample(
    minority,
    replace=True,                   # sample with replacement (bootstrap)
    n_samples=len(majority),        # bring minority up to majority count
    random_state=42
)

# Combine and shuffle so the model does not see patterns based on row order
train_balanced = pd.concat([majority, minority_upsampled]).sample(
    frac=1, random_state=42
)

X_train_bal = train_balanced.drop(columns=['is_fraud'])
y_train_bal = train_balanced['is_fraud']

print(f'\nAfter oversampling (training set):')
print(f'  Legitimate : {(y_train_bal == 0).sum():,}')
print(f'  Fraud      : {(y_train_bal == 1).sum():,}')
print(f'  Total rows : {len(y_train_bal):,}')

# ── Visualise class balance before vs. after ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before
axes[0].bar(
    ['Legitimate', 'Fraud'],
    [len(majority), len(minority)],
    color=['steelblue', 'crimson'],
    edgecolor='white'
)
axes[0].set_title('Before Oversampling (Training Set)', fontsize=12)
axes[0].set_ylabel('Count')
for i, v in enumerate([len(majority), len(minority)]):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# After
axes[1].bar(
    ['Legitimate', 'Fraud'],
    [(y_train_bal == 0).sum(), (y_train_bal == 1).sum()],
    color=['steelblue', 'crimson'],
    edgecolor='white'
)
axes[1].set_title('After Oversampling (Training Set)', fontsize=12)
axes[1].set_ylabel('Count')
for i, v in enumerate([(y_train_bal == 0).sum(), (y_train_bal == 1).sum()]):
    axes[1].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

plt.suptitle('Class Distribution: Before vs. After Oversampling',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('class_balance_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nClass imbalance handled.')
print('Test set was NOT resampled — it reflects the real-world distribution.')
print('X_train_bal and y_train_bal are ready for model training.')

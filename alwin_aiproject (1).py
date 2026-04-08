# =============================================================================
#  Banking Fraud Detection – VUIP111 Major Project
#  File     : alwin.py
#  Author   : Alwin
#  Section  : Data Cleaning & Preprocessing | Feature Scaling
# =============================================================================


# =============================================================================
#  DATA CLEANING & PREPROCESSING
# =============================================================================
#
#  Raw data is almost never model-ready. This section handles everything
#  that needs to happen between "loading the CSV" and "training the model".
#
#  Steps performed here:
#
#  Step A – Convert data types
#            transaction_date is stored as a plain string in the CSV.
#            We parse it into a proper datetime object so we can extract
#            day, month, and weekday features later in siddharth.py.
#            This step must happen BEFORE we drop the column.
#
#  Step B – Drop non-predictive columns
#            transaction_id and customer_id are just unique identifiers —
#            they carry no signal for fraud detection and would only confuse
#            the model. city is also dropped because it introduces too many
#            categories with limited benefit after encoding.
#
#  Step C – Handle missing values
#            Even though our dataset has no nulls, we write a general handler
#            so the pipeline does not crash if the dataset is updated.
#            Numeric nulls → filled with the column median (robust to outliers)
#            Categorical nulls → filled with the most frequent value (mode)
#
#  Step D – Label-encode categorical columns
#            ML models require numeric input. Columns like gender, account_type,
#            transaction_type, merchant_category, and device_type contain
#            text labels that we convert to integers using LabelEncoder.
#            We store each encoder in a dictionary (label_encoders) so we can
#            apply the SAME mapping at inference time in the Gradio app.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)

# df should be in memory from ananjan.py.
# Uncomment if running this file standalone:
# df = pd.read_csv('banking_fraud_dataset_50k.csv')


# ── Step A: Convert transaction_date to datetime ──────────────────────────────
#
#  pd.to_datetime() is smarter than manual parsing — it handles multiple
#  date formats automatically. We do this before anything else because
#  the datetime column is needed for feature extraction in siddharth.py.

print('=' * 55)
print('  STEP A : Convert transaction_date to datetime')
print('=' * 55)

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
print(f'transaction_date dtype : {df["transaction_date"].dtype}')
print(f'Sample values          : {df["transaction_date"].head(3).tolist()}')


# ── Step B: Drop non-predictive columns ──────────────────────────────────────
#
#  Columns dropped and why:
#    transaction_id → unique per row, no pattern a model can learn
#    customer_id    → unique identifier, not a feature
#    city           → too many unique values; district-level data rarely
#                     adds meaningful signal after other location-related
#                     features are present

print('\n' + '=' * 55)
print('  STEP B : Drop non-predictive columns')
print('=' * 55)

drop_cols = ['transaction_id', 'customer_id', 'city']
df_clean  = df.drop(columns=drop_cols)

print(f'Columns dropped    : {drop_cols}')
print(f'Remaining columns  : {df_clean.shape[1]}')
print(f'Dataset shape      : {df_clean.shape}')


# ── Step C: Handle missing values ─────────────────────────────────────────────
#
#  We check every column. If a column has any nulls, we fill them using a
#  strategy appropriate to the data type.
#
#  Median (not mean) is used for numeric columns because fraud amounts can
#  be highly skewed — a single extreme value would pull the mean far from
#  the typical value.

print('\n' + '=' * 55)
print('  STEP C : Handle missing values')
print('=' * 55)

null_counts = df_clean.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if len(cols_with_nulls) == 0:
    print('No missing values detected — no imputation needed.')
else:
    for col in cols_with_nulls.index:
        if df_clean[col].dtype in ['float64', 'int64']:
            fill_val = df_clean[col].median()
            df_clean[col].fillna(fill_val, inplace=True)
            print(f'  {col} (numeric)     → filled with median = {fill_val:.2f}')
        else:
            fill_val = df_clean[col].mode()[0]
            df_clean[col].fillna(fill_val, inplace=True)
            print(f'  {col} (categorical) → filled with mode = "{fill_val}"')

print('Missing value handling complete.')


# ── Step D: Label-encode categorical columns ──────────────────────────────────
#
#  select_dtypes(include=['object']) picks up any column with text values.
#  We explicitly exclude datetime columns since they are handled separately.
#
#  Each LabelEncoder is saved into a dictionary so that at inference time
#  (in the Gradio app) we can call le.transform([user_input]) and get
#  the exact same integer the model was trained on.

print('\n' + '=' * 55)
print('  STEP D : Label-encode categorical columns')
print('=' * 55)

cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f'Categorical columns found: {cat_cols}')

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f'  Encoded "{col}"  →  classes: {list(le.classes_)}')

print(f'\nTotal categorical columns encoded : {len(cat_cols)}')
print(f'Dataset shape after encoding      : {df_clean.shape}')
print('\nFirst 3 rows after preprocessing:')
print(df_clean.head(3))


# ── Visual check: encoded category distributions ──────────────────────────────
#
#  After encoding, we do a quick bar-chart check to make sure the category
#  counts look reasonable and nothing got collapsed or inflated.

if len(cat_cols) > 0:
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        df_clean[col].value_counts().plot(kind='bar', ax=ax, color='steelblue',
                                          edgecolor='white')
        ax.set_title(f'Encoded: {col}', fontsize=10)
        ax.set_xlabel('Encoded Value')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)

    plt.suptitle('Category Distributions After Label Encoding',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_category_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Category distribution plot saved.')

print('\nData cleaning & preprocessing complete.')
print(f'df_clean is ready for feature engineering (siddharth.py).')


# =============================================================================
#  FEATURE SCALING
# =============================================================================
#
#  Why scale?
#  ──────────
#  The features in our dataset span very different numeric ranges:
#    • age              → 18 to 80
#    • amount           → 1 to 200,000+
#    • account_balance  → 0 to 500,000+
#    • login_attempts   → 1 to 10
#    • distance_from_home_km → 0 to 1,000+
#
#  Without scaling, the model's distance calculations and regularisation
#  would be dominated by large-magnitude features (like amount) at the
#  expense of small-but-meaningful features (like login_attempts).
#
#  Method: StandardScaler
#  ──────────────────────
#  Transforms each feature to zero mean and unit variance:
#
#      z = (x − mean) / std
#
#  This does NOT change the shape of the distribution — it just shifts and
#  rescales so all features live on a comparable numeric scale.
#
#  Data leakage rule (strictly enforced here):
#  ────────────────────────────────────────────
#  We fit the scaler ONLY on the training data, then use the same fitted
#  scaler to transform both the training set and the test set.
#
#  Fitting on the test set would expose test-set statistics (mean, std)
#  to the model during training — a form of data leakage that makes
#  evaluation results optimistic and unreliable.
# =============================================================================

print('\n' + '=' * 55)
print('  FEATURE SCALING')
print('=' * 55)

# X_train_bal and X_test come from ananjan.py
# Uncomment and load from disk if running standalone:
# X_train_bal = pd.read_csv('X_train_bal.csv')
# X_test      = pd.read_csv('X_test.csv')

scaler = StandardScaler()

scale_cols = X_train_bal.columns.tolist()
print(f'Number of features to scale : {len(scale_cols)}')

# Fit on training data ONLY, then transform both sets
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrames so column names are preserved
X_train_scaled = pd.DataFrame(X_train_scaled, columns=scale_cols)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=scale_cols)

print(f'\nScaler fitted on {X_train_bal.shape[0]:,} training samples.')
print(f'Training set scaled  →  shape: {X_train_scaled.shape}')
print(f'Test set scaled      →  shape: {X_test_scaled.shape}')

# Show before vs after scaling for a sample feature
sample_feature = 'amount'
print(f'\nSample feature "{sample_feature}" — before vs. after scaling:')
print(f'  Before → mean: {X_train_bal[sample_feature].mean():.2f}  '
      f'std: {X_train_bal[sample_feature].std():.2f}')
print(f'  After  → mean: {X_train_scaled[sample_feature].mean():.4f}  '
      f'std: {X_train_scaled[sample_feature].std():.4f}')

# Visual: compare scaled distribution of a key feature
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(X_train_bal['amount'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('amount — Before Scaling', fontsize=11)
axes[0].set_xlabel('Amount (₹)')
axes[0].set_ylabel('Frequency')

axes[1].hist(X_train_scaled['amount'], bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('amount — After StandardScaler', fontsize=11)
axes[1].set_xlabel('Scaled Value (z-score)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Effect of StandardScaler on the "amount" Feature',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('scaling_before_after.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nFeature scaling complete.')
print('X_train_scaled and X_test_scaled are ready for model training (palak.py).')
print(f'\nFirst scaled row (sample):')
print(X_train_scaled.iloc[0].round(4))

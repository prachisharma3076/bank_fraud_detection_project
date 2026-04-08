# =============================================================================
#  Banking Fraud Detection – VUIP111 Major Project
#  File     : prachi.py
#  Author   : Prachi
#  Section  : Exploratory Data Analysis (EDA) | Save Model Artefacts | Gradio Frontend
# =============================================================================


# =============================================================================
#  EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
#
#  EDA is the foundation of any honest ML project. Before writing a single
#  line of model code, we need to understand what the data actually looks like.
#
#  What we check here:
#    • Data types         → are columns the right type (int, float, object)?
#    • Statistical summary → mean, std, min, max, percentiles for numeric cols
#    • Missing values     → which columns have nulls and how many
#    • Class distribution → how skewed is the fraud/legitimate split?
#    • Feature distributions → do fraud and legitimate transactions behave
#                              differently on key numeric features?
#    • Correlation matrix → which features are strongly related to each other
#                           or to the target (is_fraud)?
#    • Category-level patterns → fraud rates by transaction type, device,
#                                account type, city, hour of day, etc.
#
#  All plots are saved as PNG files so they can be embedded in the project
#  documentation / report.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import gradio as gr
from datetime import datetime

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)

# Load raw dataset (df should already be in memory if running after ananjan.py)
# Uncomment the line below if running this file standalone:
# df = pd.read_csv('banking_fraud_dataset_50k.csv')


# ── 1. Data Types ─────────────────────────────────────────────────────────────
print('=' * 50)
print('  DATA TYPES')
print('=' * 50)
print(df.dtypes)


# ── 2. Statistical Summary ────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  STATISTICAL SUMMARY')
print('=' * 50)
print(df.describe().round(2))


# ── 3. Missing Values ─────────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  MISSING VALUES PER COLUMN')
print('=' * 50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print('No missing values found across any column.')
else:
    print(missing[missing > 0])


# ── 4. Class Distribution ─────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  TARGET CLASS DISTRIBUTION')
print('=' * 50)
fraud_counts = df['is_fraud'].value_counts()
fraud_pct    = df['is_fraud'].value_counts(normalize=True) * 100
print(pd.DataFrame({'Count': fraud_counts, 'Percentage (%)': fraud_pct.round(2)}))

# Plot: bar + pie side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(
    ['Legitimate (0)', 'Fraud (1)'],
    fraud_counts.values,
    color=['steelblue', 'crimson'],
    edgecolor='white', linewidth=1.2
)
axes[0].set_title('Transaction Class Counts', fontsize=12)
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(
    fraud_counts.values,
    labels=['Legitimate', 'Fraud'],
    autopct='%1.1f%%',
    colors=['steelblue', 'crimson'],
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
)
axes[1].set_title('Fraud vs. Legitimate (%)', fontsize=12)

plt.suptitle('Class Imbalance Overview', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Class distribution chart saved.')


# ── 5. Feature Distributions: Fraud vs. Legitimate ───────────────────────────
numeric_cols = [
    'age', 'amount', 'account_balance',
    'transaction_hour', 'login_attempts', 'distance_from_home_km'
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(
        df[df['is_fraud'] == 0][col], bins=40, alpha=0.6,
        color='steelblue', label='Legitimate', density=True
    )
    axes[i].hist(
        df[df['is_fraud'] == 1][col], bins=40, alpha=0.6,
        color='crimson', label='Fraud', density=True
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)

plt.suptitle('Feature Distributions: Fraud vs. Legitimate',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Feature distribution plot saved.')


# ── 6. Correlation Heatmap ────────────────────────────────────────────────────
#
#  We include the binary flags and the target variable in the correlation
#  matrix so we can see which numeric features are most directly linked to fraud.

corr_cols = numeric_cols + ['is_night_transaction', 'is_weekend', 'is_new_device', 'is_fraud']

plt.figure(figsize=(12, 8))
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show only lower triangle

sns.heatmap(
    corr_matrix, mask=mask,
    annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Correlation heatmap saved.')


# ── 7. Fraud Rate by Category ─────────────────────────────────────────────────
#
#  For each categorical column we compute the average fraud rate per category.
#  This reveals which groups (e.g., device type, transaction type) are
#  disproportionately associated with fraudulent activity.

cat_fraud_cols = ['transaction_type', 'device_type', 'account_type', 'merchant_category']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(cat_fraud_cols):
    fraud_rate = df.groupby(col)['is_fraud'].mean().mul(100).sort_values(ascending=False)
    axes[i].bar(
        fraud_rate.index, fraud_rate.values,
        color='crimson', edgecolor='white', alpha=0.85
    )
    axes[i].set_title(f'Fraud Rate by {col} (%)', fontsize=11)
    axes[i].set_ylabel('Fraud Rate (%)')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=15)

plt.suptitle('Fraud Rate Across Categories', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_fraud_by_category.png', dpi=150, bbox_inches='tight')
plt.show()
print('Category fraud-rate charts saved.')


# ── 8. Fraud Rate by Hour of Day ─────────────────────────────────────────────
fraud_by_hour = df.groupby('transaction_hour')['is_fraud'].mean().mul(100)

plt.figure(figsize=(12, 4))
plt.plot(
    fraud_by_hour.index, fraud_by_hour.values,
    color='crimson', linewidth=2, marker='o', markersize=4
)
plt.fill_between(fraud_by_hour.index, fraud_by_hour.values, alpha=0.15, color='crimson')
plt.title('Fraud Rate by Hour of Day', fontsize=13, fontweight='bold')
plt.xlabel('Hour  (0 = midnight  |  12 = noon)')
plt.ylabel('Fraud Rate (%)')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('eda_fraud_by_hour.png', dpi=150, bbox_inches='tight')
plt.show()
print('Hourly fraud-rate chart saved.')

print('\nEDA complete. All plots have been saved as PNG files.')


# =============================================================================
#  SAVE MODEL & ARTEFACTS FOR DEPLOYMENT
# =============================================================================
#
#  Once the model is trained and evaluated (palak.py / siddharth.py),
#  we serialise everything needed to reproduce predictions in the backend.
#
#  Files saved:
#    best_model.pkl       → the trained Random Forest (tuned by GridSearchCV)
#    scaler.pkl           → the fitted StandardScaler (same transform at inference)
#    label_encoders.pkl   → LabelEncoder objects for each categorical column
#    model_metadata.json  → human-readable summary of the model and its performance
#
#  Why save all of these?
#  ────────────────────────
#  At inference time (when a user submits a transaction via the web app),
#  we must apply EXACTLY the same preprocessing steps as during training:
#    1. Encode categorical values with the same LabelEncoder mappings
#    2. Apply the same StandardScaler fitted on the training data
#    3. Pass the result through the model in the exact feature order
#
#  If any of these steps differ, predictions will be wrong or crash.
# =============================================================================

os.makedirs('model_artefacts', exist_ok=True)

# Save trained model
joblib.dump(best_model, 'model_artefacts/best_model.pkl')
print('Saved: model_artefacts/best_model.pkl')

# Save scaler
joblib.dump(scaler, 'model_artefacts/scaler.pkl')
print('Saved: model_artefacts/scaler.pkl')

# Save label encoders
joblib.dump(label_encoders, 'model_artefacts/label_encoders.pkl')
print('Saved: model_artefacts/label_encoders.pkl')

# Save metadata as JSON (human-readable, useful for documentation)
metadata = {
    'model_type'      : 'RandomForestClassifier',
    'best_params'     : grid_search.best_params_,
    'feature_columns' : X_train_bal.columns.tolist(),
    'categorical_cols': cat_cols,
    'target_column'   : 'is_fraud',
    'performance': {
        'accuracy' : round(acc, 4),
        'precision': round(precision, 4),
        'recall'   : round(recall, 4),
        'f1_score' : round(f1, 4),
        'roc_auc'  : round(roc_auc, 4)
    }
}

with open('model_artefacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print('Saved: model_artefacts/model_metadata.json')

print('\n=== Final Model Performance ===')
for k, v in metadata['performance'].items():
    print(f'  {k:10s}: {v}')

print('\nAll artefacts saved. Model is ready for deployment.')


# =============================================================================
#  GRADIO FRONTEND
# =============================================================================
#
#  Gradio lets us build an interactive web UI with pure Python — no HTML,
#  CSS, or JavaScript needed. Users fill in a form, click a button,
#  and see the fraud prediction result instantly.
#
#  How it works end-to-end:
#    User fills form  →  predict_fraud() runs  →  model returns probability
#    →  result displayed as a formatted markdown card + numeric gauge
#
#  HOW TO RUN:
#    1. Ensure model_artefacts/ folder exists in the same directory
#    2. pip install gradio joblib scikit-learn pandas numpy
#    3. python prachi.py
#    4. Open http://localhost:7860 in your browser
# =============================================================================

# ── Load saved artefacts ──────────────────────────────────────────────────────
print('Loading model artefacts...')
model          = joblib.load('model_artefacts/best_model.pkl')
scaler_inf     = joblib.load('model_artefacts/scaler.pkl')
label_encoders = joblib.load('model_artefacts/label_encoders.pkl')

with open('model_artefacts/model_metadata.json') as f:
    metadata = json.load(f)

FEATURE_COLS = metadata['feature_columns']
print('All artefacts loaded.')

# ── Dropdown options (must match LabelEncoder classes from training) ───────────
GENDER_OPTS   = ['Female', 'Male']
ACCOUNT_OPTS  = ['Credit', 'Current', 'Savings']
TXN_OPTS      = ['ATM', 'Card', 'IMPS', 'NEFT', 'Net Banking', 'UPI']
MERCHANT_OPTS = ['Dining', 'Education', 'Electronics', 'Entertainment',
                 'Fuel', 'Groceries', 'Healthcare', 'Shopping', 'Travel', 'Utilities']
DEVICE_OPTS   = ['ATM', 'Desktop', 'Mobile', 'POS']


# ── Prediction function ───────────────────────────────────────────────────────
def predict_fraud(
    age, gender, account_type, transaction_type, merchant_category,
    amount, account_balance, transaction_date, transaction_hour,
    is_night_transaction, is_weekend, device_type,
    login_attempts, is_new_device, distance_from_home_km
):
    try:
        dt = datetime.strptime(str(transaction_date), '%Y-%m-%d')

        # Encode categoricals using the saved LabelEncoders
        gender_enc   = label_encoders['gender'].transform([gender])[0]
        account_enc  = label_encoders['account_type'].transform([account_type])[0]
        txn_enc      = label_encoders['transaction_type'].transform([transaction_type])[0]
        merchant_enc = label_encoders['merchant_category'].transform([merchant_category])[0]
        device_enc   = label_encoders['device_type'].transform([device_type])[0]

        # Engineered features — must exactly mirror what siddharth.py computed
        amount_to_balance_ratio = amount / (account_balance if account_balance != 0 else 0.01)
        transaction_day         = dt.day
        transaction_month       = dt.month
        day_of_week             = dt.weekday()
        high_amount_flag        = int(amount > 55151.00)   # 95th percentile from training
        multi_login_flag        = int(login_attempts > 1)

        # Build input row in the exact feature order the model was trained on
        row = {
            'age'                    : age,
            'gender'                 : gender_enc,
            'account_type'           : account_enc,
            'transaction_type'       : txn_enc,
            'merchant_category'      : merchant_enc,
            'amount'                 : amount,
            'account_balance'        : account_balance,
            'transaction_hour'       : transaction_hour,
            'is_night_transaction'   : int(is_night_transaction),
            'is_weekend'             : int(is_weekend),
            'device_type'            : device_enc,
            'login_attempts'         : login_attempts,
            'is_new_device'          : int(is_new_device),
            'distance_from_home_km'  : distance_from_home_km,
            'amount_to_balance_ratio': amount_to_balance_ratio,
            'transaction_day'        : transaction_day,
            'transaction_month'      : transaction_month,
            'day_of_week'            : day_of_week,
            'high_amount_flag'       : high_amount_flag,
            'multi_login_flag'       : multi_login_flag,
        }

        input_df     = pd.DataFrame([row])[FEATURE_COLS]
        input_scaled = scaler_inf.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]

        fraud_prob = round(float(proba[1]) * 100, 2)
        legit_prob = round(float(proba[0]) * 100, 2)

        # Risk label
        if fraud_prob >= 70:
            risk = '🔴 HIGH RISK'
        elif fraud_prob >= 40:
            risk = '🟡 MEDIUM RISK'
        else:
            risk = '🟢 LOW RISK'

        # Output card
        if prediction == 1:
            verdict = '🚨 FRAUDULENT TRANSACTION DETECTED'
            action  = 'Block transaction and notify the customer immediately.'
        else:
            verdict = '✅ LEGITIMATE TRANSACTION'
            action  = 'Allow transaction to proceed.'

        summary = (
            f'### {verdict}\n\n'
            f'**Confidence:** {fraud_prob if prediction == 1 else legit_prob}%\n\n'
            f'**Risk Level:** {risk}\n\n'
            f'**Recommended Action:** {action}\n\n'
            f'---\n'
            f'**Probability Breakdown**\n'
            f'- Legitimate : {legit_prob}%\n'
            f'- Fraud       : {fraud_prob}%\n\n'
            f'**Key Engineered Signals**\n'
            f'- Amount-to-Balance Ratio : {amount_to_balance_ratio:.4f}\n'
            f'- High Amount Flag        : {"Yes" if high_amount_flag else "No"} '
            f'(threshold ₹55,151)\n'
            f'- Multi-Login Flag        : {"Yes" if multi_login_flag else "No"}\n'
            f'- Day of Week             : '
            f'{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week]}\n'
        )

        return summary, fraud_prob

    except Exception as e:
        return f'Error during prediction: {str(e)}', 0.0


# ── Gradio UI Layout ──────────────────────────────────────────────────────────
with gr.Blocks(
    title='Banking Fraud Detection',
    theme=gr.themes.Soft(primary_hue='blue', secondary_hue='red'),
) as demo:

    gr.Markdown(
        '''
        # 🏦 Banking Fraud Detection System
        ### AI-Powered Transaction Risk Analysis — VUIP111 Major Project
        Fill in the transaction details below and click **Analyse Transaction**.
        ---
        '''
    )

    with gr.Row():

        # Left: Customer & Account
        with gr.Column(scale=1):
            gr.Markdown('### 👤 Customer & Account Details')
            age             = gr.Slider(18, 80, value=35, step=1, label='Customer Age')
            gender          = gr.Radio(choices=GENDER_OPTS, value='Male', label='Gender')
            account_type    = gr.Dropdown(choices=ACCOUNT_OPTS, value='Savings', label='Account Type')
            account_balance = gr.Number(value=50000, label='Account Balance (₹)', minimum=0)

        # Middle: Transaction Details
        with gr.Column(scale=1):
            gr.Markdown('### 💳 Transaction Details')
            amount           = gr.Number(value=1500, label='Transaction Amount (₹)', minimum=0)
            transaction_type = gr.Dropdown(choices=TXN_OPTS, value='UPI', label='Transaction Type')
            merchant_cat     = gr.Dropdown(choices=MERCHANT_OPTS, value='Groceries', label='Merchant Category')
            txn_date         = gr.DateTime(label='Transaction Date', include_time=False,
                                           value='2024-01-15', type='string')
            txn_hour         = gr.Slider(0, 23, value=12, step=1, label='Transaction Hour (0–23)')

        # Right: Device & Behaviour
        with gr.Column(scale=1):
            gr.Markdown('### 📱 Device & Behaviour Signals')
            device_type    = gr.Dropdown(choices=DEVICE_OPTS, value='Mobile', label='Device Type')
            login_attempts = gr.Slider(1, 10, value=1, step=1, label='Login Attempts Before Transaction')
            distance       = gr.Number(value=15.0, label='Distance from Home (km)', minimum=0)
            is_night       = gr.Checkbox(label='🌙 Night Transaction (10 PM – 6 AM)', value=False)
            is_weekend     = gr.Checkbox(label='📅 Weekend Transaction', value=False)
            is_new_device  = gr.Checkbox(label='📲 New / Unrecognised Device', value=False)

    with gr.Row():
        analyse_btn = gr.Button('🔍 Analyse Transaction', variant='primary', size='lg')

    with gr.Row():
        with gr.Column(scale=2):
            result_md = gr.Markdown(label='Analysis Result')
        with gr.Column(scale=1):
            fraud_gauge = gr.Number(label='Fraud Probability (%)', interactive=False)

    gr.Markdown('---\n### 📋 Try Example Transactions')
    gr.Examples(
        examples=[
            # Low risk – small UPI grocery, daytime, known device
            [35, 'Male',   'Savings', 'UPI',        'Groceries',   500,   55000, '2024-03-10', 11, False, False, 'Mobile',  1, False, 5.0],
            # High risk – large amount, 2 AM, new device, multiple logins
            [42, 'Female', 'Credit',  'Net Banking', 'Electronics', 98000, 12000, '2024-03-10', 2,  True,  False, 'Desktop', 4, True,  180.0],
            # Medium risk – ATM, weekend, late night, far from home
            [28, 'Male',   'Current', 'ATM',         'Shopping',    15000, 30000, '2024-03-09', 22, True,  True,  'ATM',     2, False, 95.0],
        ],
        inputs=[
            age, gender, account_type, transaction_type, merchant_cat,
            amount, account_balance, txn_date, txn_hour,
            is_night, is_weekend, device_type,
            login_attempts, is_new_device, distance
        ],
        label='Click any row to auto-fill the form'
    )

    gr.Markdown(
        '\n---\n'
        '*Model: Random Forest Classifier  ·  '
        'Dataset: 50,000 banking transactions  ·  Course: VUIP111*'
    )

    analyse_btn.click(
        fn=predict_fraud,
        inputs=[
            age, gender, account_type, transaction_type, merchant_cat,
            amount, account_balance, txn_date, txn_hour,
            is_night, is_weekend, device_type,
            login_attempts, is_new_device, distance
        ],
        outputs=[result_md, fraud_gauge]
    )


if __name__ == '__main__':
    demo.launch(share=False)   # set share=True for a public Gradio link

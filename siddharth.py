# =============================================================================
#  Banking Fraud Detection – VUIP111 Major Project
#  File     : siddharth.py
#  Author   : Siddharth
#  Section  : Feature Engineering | Model Evaluation
# =============================================================================


# =============================================================================
#  FEATURE ENGINEERING
# =============================================================================
#
#  Raw features alone rarely give a model the full picture. Feature engineering
#  is the process of creating new, more informative columns from the data we
#  already have — using domain knowledge to highlight patterns that are hard
#  for the model to discover on its own.
#
#  For banking fraud detection, we know from real-world fraud patterns that:
#    • A transaction of ₹90,000 from an account with ₹500 balance is riskier
#      than the same amount from an account with ₹500,000.
#    • Fraudsters often strike at odd hours (2 AM) on weekends.
#    • Multiple failed login attempts before a transaction are a red flag.
#    • Unusually large transactions (top 5% of amounts) warrant extra scrutiny.
#
#  We translate these domain insights into concrete numeric features below.
#
#  New features created:
#  ─────────────────────
#  amount_to_balance_ratio → transaction amount ÷ account balance
#                            flags transactions that are disproportionately
#                            large relative to the customer's funds
#
#  transaction_day         → day of the month (1–31) extracted from date
#  transaction_month       → month (1–12) extracted from date
#  day_of_week             → 0 = Monday, 6 = Sunday
#                            captures weekend / weekday patterns
#
#  high_amount_flag        → 1 if amount is above the 95th percentile,
#                            0 otherwise; a hard binary signal for very
#                            large transactions
#
#  multi_login_flag        → 1 if more than one login attempt was made
#                            before the transaction; 0 otherwise
#
#  After extraction, transaction_date is dropped — its information is now
#  fully captured in the three derived temporal features above.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)

# df_clean comes from alwin.py.
# Uncomment if running standalone:
# df_clean = pd.read_csv('df_clean_preprocessed.csv')
# df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'])


# ── Feature 1: Amount-to-Balance Ratio ───────────────────────────────────────
#
#  A large transaction is only suspicious in context. ₹50,000 is normal for
#  a high-net-worth customer but alarming if their balance is ₹800.
#  This ratio captures that context.
#
#  We replace zero balances with a tiny positive number to avoid division
#  by zero (which would produce inf and crash the model).

print('=' * 55)
print('  FEATURE 1 : Amount-to-Balance Ratio')
print('=' * 55)

df_clean['amount_to_balance_ratio'] = (
    df_clean['amount'] / df_clean['account_balance'].replace(0, 0.01)
)

print(f'Mean ratio (Legitimate) : '
      f'{df_clean[df_clean["is_fraud"] == 0]["amount_to_balance_ratio"].mean():.4f}')
print(f'Mean ratio (Fraud)      : '
      f'{df_clean[df_clean["is_fraud"] == 1]["amount_to_balance_ratio"].mean():.4f}')
print('Higher ratio in fraud rows confirms this is a useful signal.')


# ── Feature 2: Temporal Features from transaction_date ───────────────────────
#
#  The raw date column holds three pieces of timing information that can be
#  independently predictive. We extract each one as a separate numeric column.

print('\n' + '=' * 55)
print('  FEATURE 2 : Temporal Features (day / month / weekday)')
print('=' * 55)

df_clean['transaction_day']   = df_clean['transaction_date'].dt.day
df_clean['transaction_month'] = df_clean['transaction_date'].dt.month
df_clean['day_of_week']       = df_clean['transaction_date'].dt.dayofweek  # 0=Mon, 6=Sun

print('transaction_day   → range:', df_clean['transaction_day'].min(),
      'to', df_clean['transaction_day'].max())
print('transaction_month → range:', df_clean['transaction_month'].min(),
      'to', df_clean['transaction_month'].max())
print('day_of_week       → 0=Mon ... 6=Sun')

# Quick check: is there a fraud spike on certain days of the week?
fraud_by_dow = df_clean.groupby('day_of_week')['is_fraud'].mean().mul(100)
print('\nFraud rate by day of week (%):')
print(fraud_by_dow.round(2).rename({0:'Mon',1:'Tue',2:'Wed',
                                    3:'Thu',4:'Fri',5:'Sat',6:'Sun'}))


# ── Feature 3: High-Amount Flag ───────────────────────────────────────────────
#
#  Instead of relying on the model to discover the threshold itself,
#  we explicitly tell it: "this is an unusually large transaction."
#  The threshold is the 95th percentile of amount in the full dataset.

print('\n' + '=' * 55)
print('  FEATURE 3 : High-Amount Flag (top 5% of transactions)')
print('=' * 55)

high_amount_threshold       = df_clean['amount'].quantile(0.95)
df_clean['high_amount_flag'] = (df_clean['amount'] > high_amount_threshold).astype(int)

print(f'95th percentile threshold : ₹{high_amount_threshold:,.2f}')
print(f'Transactions flagged      : {df_clean["high_amount_flag"].sum():,} '
      f'({df_clean["high_amount_flag"].mean() * 100:.1f}% of dataset)')
print(f'Fraud rate among flagged  : '
      f'{df_clean[df_clean["high_amount_flag"] == 1]["is_fraud"].mean() * 100:.2f}%')
print(f'Fraud rate among normal   : '
      f'{df_clean[df_clean["high_amount_flag"] == 0]["is_fraud"].mean() * 100:.2f}%')


# ── Feature 4: Multi-Login Flag ───────────────────────────────────────────────
#
#  Multiple failed login attempts before completing a transaction is a
#  classic indicator of credential stuffing or brute-force attacks.
#  We binarise this signal: 1 if login_attempts > 1, else 0.

print('\n' + '=' * 55)
print('  FEATURE 4 : Multi-Login Flag (login_attempts > 1)')
print('=' * 55)

df_clean['multi_login_flag'] = (df_clean['login_attempts'] > 1).astype(int)

print(f'Fraud rate — single login  : '
      f'{df_clean[df_clean["multi_login_flag"] == 0]["is_fraud"].mean() * 100:.2f}%')
print(f'Fraud rate — multiple login: '
      f'{df_clean[df_clean["multi_login_flag"] == 1]["is_fraud"].mean() * 100:.2f}%')


# ── Drop raw date column ──────────────────────────────────────────────────────
#
#  Now that we have extracted day, month, and weekday, the original date
#  column is redundant. Models cannot use datetime objects directly anyway,
#  so we drop it before proceeding to the split.

df_clean.drop(columns=['transaction_date'], inplace=True)

print('\n' + '=' * 55)
print('  SUMMARY : Feature Engineering Complete')
print('=' * 55)
new_feats = ['amount_to_balance_ratio', 'transaction_day',
             'transaction_month', 'day_of_week',
             'high_amount_flag', 'multi_login_flag']
print(f'New features added  : {new_feats}')
print(f'Final dataset shape : {df_clean.shape}')
print(df_clean.head(3))


# ── Visualise engineered features ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(new_feats):
    if df_clean[feat].nunique() <= 2:
        # Binary flag → bar chart
        fraud_rate = df_clean.groupby(feat)['is_fraud'].mean().mul(100)
        axes[i].bar(fraud_rate.index.astype(str), fraud_rate.values,
                    color=['steelblue', 'crimson'], edgecolor='white')
        axes[i].set_title(f'Fraud Rate by {feat}', fontsize=10)
        axes[i].set_ylabel('Fraud Rate (%)')
    else:
        # Continuous feature → overlapping histogram
        axes[i].hist(df_clean[df_clean['is_fraud'] == 0][feat], bins=30,
                     alpha=0.6, color='steelblue', label='Legit', density=True)
        axes[i].hist(df_clean[df_clean['is_fraud'] == 1][feat], bins=30,
                     alpha=0.6, color='crimson', label='Fraud', density=True)
        axes[i].set_title(f'Distribution of {feat}', fontsize=10)
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
    axes[i].set_xlabel(feat)

plt.suptitle('Engineered Features: Fraud vs. Legitimate', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_engineering_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print('Feature engineering plots saved.')


# =============================================================================
#  MODEL EVALUATION
# =============================================================================
#
#  Once the model is trained (palak.py), we evaluate it on the held-out test
#  set — data the model has NEVER seen during training or tuning.
#
#  Why multiple metrics?
#  ─────────────────────
#  Accuracy alone is misleading on imbalanced datasets. If only 3.5% of
#  transactions are fraud, a model predicting "Legitimate" every time still
#  achieves 96.5% accuracy — while catching zero frauds.
#
#  We use:
#  ─────────────────────────────────────────────────────────────────────────────
#  Accuracy   → % of all predictions that were correct
#               (context: useful but insufficient alone here)
#
#  Precision  → of all transactions predicted as fraud, what fraction were
#               actually fraud?
#               (context: high precision = fewer false alarms for customers)
#
#  Recall     → of all actual fraud transactions, what fraction did we catch?
#               (context: THIS is the most critical metric for fraud detection.
#                A missed fraud = financial loss. We want recall as high as
#                possible, even if it costs us some precision.)
#
#  F1-Score   → harmonic mean of precision and recall.
#               A single number that balances both. Used as the optimisation
#               target in GridSearchCV (palak.py).
#
#  ROC-AUC    → measures the model's ability to rank fraud transactions above
#               legitimate ones across all probability thresholds.
#               AUC = 1.0 → perfect; AUC = 0.5 → random guessing.
#
#  Confusion Matrix:
#  ─────────────────
#  TP (top-right)  → correctly identified fraud          ← we want this high
#  FP (top-left)   → legitimate flagged as fraud         ← customer inconvenience
#  FN (bottom-right) → fraud missed entirely             ← financial loss, worst case
#  TN (bottom-left)  → correctly cleared as legitimate   ← we want this high
# =============================================================================

print('\n' + '=' * 55)
print('  MODEL EVALUATION')
print('=' * 55)

# best_model, X_test_scaled, y_test come from palak.py and alwin.py
# Uncomment if running standalone after loading artefacts:
# import joblib
# best_model    = joblib.load('model_artefacts/best_model.pkl')

y_pred      = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f'\n  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)')
print(f'  Precision : {precision:.4f}')
print(f'  Recall    : {recall:.4f}')
print(f'  F1-Score  : {f1:.4f}')
print(f'  ROC-AUC   : {roc_auc:.4f}')
print()
print('Full Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))


# ── Confusion Matrix ──────────────────────────────────────────────────────────
print('=' * 55)
print('  CONFUSION MATRIX')
print('=' * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Legitimate', 'Fraud'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]
print(f'True Positives  (fraud caught)    : {TP:,}')
print(f'False Positives (false alarms)    : {FP:,}')
print(f'False Negatives (fraud missed)    : {FN:,}')
print(f'True Negatives  (legit cleared)   : {TN:,}')


# ── ROC Curve ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  ROC CURVE')
print('=' * 55)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

axes[1].plot(fpr, tpr, color='crimson', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1,
             label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='crimson')
axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

plt.suptitle('Model Performance Visualisation', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'AUC = {roc_auc:.4f}  — the model can discriminate fraud from legit '
      f'with {roc_auc * 100:.1f}% reliability across all thresholds.')


# ── Feature Importance ────────────────────────────────────────────────────────
#
#  Random Forest calculates feature importance by measuring how much each
#  feature reduces impurity (Gini index) across all decision trees.
#  A higher importance score means the feature was more useful for splitting
#  data correctly throughout the forest.
#
#  This gives us interpretability — we can explain to a bank stakeholder
#  *why* the model flagged a transaction, not just that it did.

print('\n' + '=' * 55)
print('  FEATURE IMPORTANCE')
print('=' * 55)

importances  = best_model.feature_importances_
feat_names   = X_train_bal.columns
feat_imp_df  = (
    pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    .sort_values('Importance', ascending=False)
    .reset_index(drop=True)
)

print('Top 10 most important features:')
print(feat_imp_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_imp_df.head(12),
    x='Importance', y='Feature',
    palette='viridis_r', edgecolor='white'
)
plt.title('Top 12 Feature Importances — Random Forest',
          fontsize=13, fontweight='bold')
plt.xlabel('Importance Score (Gini Impurity Reduction)')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nModel evaluation complete.')
print('Results and plots have been saved.')
print(f'\nFinal Summary:')
print(f'  Accuracy  : {acc * 100:.2f}%')
print(f'  Recall    : {recall * 100:.2f}%  ← most important for fraud detection')
print(f'  F1-Score  : {f1:.4f}')
print(f'  ROC-AUC   : {roc_auc:.4f}')

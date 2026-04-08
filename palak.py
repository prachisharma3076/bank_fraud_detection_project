# =============================================================================
#  Banking Fraud Detection – VUIP111 Major Project
#  File     : palak.py
#  Author   : Palak
#  Section  : Model Training | Hyperparameter Tuning
# =============================================================================


# =============================================================================
#  MODEL TRAINING
# =============================================================================
#
#  Algorithm chosen: Random Forest Classifier
#  ───────────────────────────────────────────
#  A Random Forest is an ensemble of decision trees. Instead of building one
#  large tree (which tends to overfit), it builds many smaller trees on random
#  subsets of the data and features, then combines their predictions by
#  majority vote.
#
#  Why Random Forest for fraud detection?
#  ───────────────────────────────────────
#  • Handles non-linear relationships between features without manual tuning
#  • Naturally robust to outliers (extreme transaction amounts won't derail it)
#  • Provides built-in feature importances so we can explain predictions
#  • Works well with the mix of numeric and encoded categorical features we have
#  • Less prone to overfitting than a single deep decision tree
#  • class_weight='balanced' adds extra penalty for misclassifying the minority
#    class (fraud), complementing the oversampling we did in ananjan.py
#
#  Key parameters used:
#  ─────────────────────
#  n_estimators=100  → number of trees; 100 balances speed and accuracy
#  class_weight='balanced' → still useful on top of oversampling to reinforce
#                            the importance of catching fraud
#  random_state=42   → makes results reproducible across runs
#  n_jobs=-1         → uses all available CPU cores for parallel training
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)

# X_train_scaled, y_train_bal come from alwin.py and ananjan.py.
# Uncomment if running standalone:
# import pandas as pd
# X_train_scaled = pd.read_csv('X_train_scaled.csv')
# y_train_bal    = pd.read_csv('y_train_bal.csv').squeeze()


# ── Build the base Random Forest ──────────────────────────────────────────────
print('=' * 55)
print('  MODEL TRAINING — Random Forest Classifier')
print('=' * 55)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print(f'Training on {X_train_scaled.shape[0]:,} balanced samples '
      f'with {X_train_scaled.shape[1]} features...')

rf_model.fit(X_train_scaled, y_train_bal)

print('\nModel training complete.')
print(f'  Trees trained     : {rf_model.n_estimators}')
print(f'  Features per tree : {rf_model.n_features_in_}')
print(f'  Classes           : {rf_model.classes_.tolist()}  '
      f'(0=Legitimate, 1=Fraud)')


# ── Cross-Validation on training set ─────────────────────────────────────────
#
#  Cross-validation gives us an early signal of how well the model
#  generalises — before we ever touch the test set.
#
#  We use 5-fold CV with F1 scoring because accuracy is misleading on
#  imbalanced data. F1 = harmonic mean of precision and recall.
#
#  If the CV scores are consistent across folds, the model is stable.
#  High variance between folds suggests overfitting.

print('\n' + '=' * 55)
print('  CROSS-VALIDATION (5-Fold, F1 scoring)')
print('=' * 55)

cv_scores = cross_val_score(
    rf_model, X_train_scaled, y_train_bal,
    cv=5, scoring='f1', n_jobs=-1
)

print(f'F1 scores per fold : {cv_scores.round(4)}')
print(f'Mean F1            : {cv_scores.mean():.4f}')
print(f'Std deviation      : {cv_scores.std():.4f}')

if cv_scores.std() < 0.01:
    print('Low variance across folds — model is stable and consistent.')
elif cv_scores.std() < 0.03:
    print('Moderate variance — acceptable, but watch for overfitting.')
else:
    print('High variance — consider more regularisation or more data.')

# Plot CV scores
plt.figure(figsize=(8, 4))
bars = plt.bar(
    [f'Fold {i+1}' for i in range(5)],
    cv_scores,
    color='steelblue', edgecolor='white'
)
plt.axhline(cv_scores.mean(), color='crimson', linestyle='--', linewidth=1.5,
            label=f'Mean F1 = {cv_scores.mean():.4f}')
for bar, score in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f'{score:.4f}', ha='center', fontsize=9, fontweight='bold')
plt.title('5-Fold Cross-Validation F1 Scores', fontsize=12, fontweight='bold')
plt.ylabel('F1 Score')
plt.ylim(0.85, 1.01)
plt.legend()
plt.tight_layout()
plt.savefig('cv_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print('Cross-validation plot saved.')


# =============================================================================
#  HYPERPARAMETER TUNING
# =============================================================================
#
#  What are hyperparameters?
#  ──────────────────────────
#  Regular parameters are learned FROM the data during training (e.g., the
#  split thresholds in each decision tree). Hyperparameters are set BY US
#  BEFORE training — they control how the model learns rather than what it
#  learns.
#
#  Examples:
#    • n_estimators   → how many trees to build
#    • max_depth      → maximum depth each tree is allowed to grow
#    • min_samples_split → minimum samples required to split an internal node
#    • max_features   → number of features considered at each split
#
#  Why tune them?
#  ───────────────
#  The default values are not always optimal for our specific dataset.
#  A forest with 100 shallow trees may underfit; 200 deep trees may overfit.
#  Tuning finds the best balance.
#
#  Method: GridSearchCV
#  ─────────────────────
#  GridSearchCV exhaustively tests EVERY possible combination from the
#  parameter grid, evaluating each with k-fold cross-validation, then
#  returns the combination with the highest mean CV score.
#
#  With our grid of 2 × 3 × 2 × 2 = 24 combinations and 3-fold CV,
#  this trains 72 models in total. We use n_jobs=-1 to parallelise this.
#
#  Scoring metric: F1 (not accuracy)
#  ───────────────────────────────────
#  We optimise for F1 because our dataset is imbalanced. Accuracy would
#  reward a model for being good at the majority class (legitimate),
#  whereas F1 equally penalises missing fraud (low recall) and
#  raising false alarms (low precision).
# =============================================================================

print('\n' + '=' * 55)
print('  HYPERPARAMETER TUNING — GridSearchCV')
print('=' * 55)

param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [None, 15, 25],
    'min_samples_split': [2, 5],
    'max_features'     : ['sqrt', 'log2']
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)

print(f'Parameter grid:')
for k, v in param_grid.items():
    print(f'  {k:20s}: {v}')
print(f'\nTotal combinations to test : {total_combos}')
print(f'CV folds                   : 3')
print(f'Total model fits           : {total_combos * 3}')
print(f'Scoring metric             : F1')
print('\nRunning GridSearchCV — this may take a few minutes...')

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train_bal)

print('\nGridSearchCV complete.')
print('\nBest parameters found:')
for k, v in grid_search.best_params_.items():
    print(f'  {k:20s}: {v}')
print(f'\nBest CV F1 score : {grid_search.best_score_:.4f}')


# ── Extract the best model ─────────────────────────────────────────────────────
#
#  grid_search.best_estimator_ is the model already re-fitted on the FULL
#  training set using the best parameters. This is what we use for all
#  downstream evaluation and deployment.

best_model = grid_search.best_estimator_
print('\nbest_model is now the tuned Random Forest — ready for evaluation.')


# ── Compare default vs tuned model ────────────────────────────────────────────
#
#  We compare the base Random Forest (default params) against the tuned one
#  on the training data to show the improvement from tuning.
#  Note: final evaluation is always done on the test set in siddharth.py.

default_score = cross_val_score(rf_model, X_train_scaled, y_train_bal,
                                 cv=3, scoring='f1', n_jobs=-1).mean()
tuned_score   = grid_search.best_score_

print('\n' + '=' * 55)
print('  DEFAULT vs. TUNED MODEL — Training CV F1')
print('=' * 55)
print(f'  Default Random Forest  : {default_score:.4f}')
print(f'  Tuned Random Forest    : {tuned_score:.4f}')
improvement = (tuned_score - default_score) * 100
if improvement > 0:
    print(f'  Improvement            : +{improvement:.2f}% F1 from tuning')
else:
    print(f'  Note: default params were already near-optimal for this dataset.')


# ── Visualise GridSearch results ──────────────────────────────────────────────
#
#  We plot the mean CV F1 score for each combination of n_estimators and
#  max_depth (averaging over min_samples_split and max_features) to see
#  which part of the grid performed best.

results_df = pd.DataFrame(grid_search.cv_results_)

pivot_data = results_df.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators',
    aggfunc='mean'
)

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot_data, annot=True, fmt='.4f',
    cmap='YlGn', linewidths=0.5,
    cbar_kws={'label': 'Mean CV F1 Score'}
)
plt.title('GridSearchCV: Mean F1 by max_depth & n_estimators',
          fontsize=12, fontweight='bold')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.savefig('gridsearch_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('GridSearch results heatmap saved.')


# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  TRAINING & TUNING SUMMARY')
print('=' * 55)
print(f'Algorithm        : Random Forest Classifier')
print(f'Training samples : {X_train_scaled.shape[0]:,} (balanced via oversampling)')
print(f'Features used    : {X_train_scaled.shape[1]}')
print(f'CV F1 (default)  : {default_score:.4f}')
print(f'CV F1 (tuned)    : {tuned_score:.4f}')
print(f'Best params      : {grid_search.best_params_}')
print('\nbest_model is now ready. Pass it to siddharth.py for full evaluation.')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import numpy as np

# Load and preprocess your data (replace with your actual data loading)
data = pd.read_csv("/content/lastfm-dataset/Last.fm_data.csv")
data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d %b %Y %H:%M')
data = data.sort_values(['Username', 'Track', 'timestamp'])
data['repeated_play'] = data.groupby(['Username', 'Track'])['timestamp'].transform(lambda x: (x.diff().dt.days <= 30).astype(int))
data['repeated_play'] = data['repeated_play'].fillna(0)
 
# Encode categorical features
label_encoders = {}
for col in ['Username', 'Track', 'Artist', 'Album']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data[['Username', 'Track', 'Artist', 'Album']]
y = data['repeated_play']

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train Random Forest with predefined hyperparameters
best_rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5,
                                 min_samples_leaf=2, random_state=42, max_features='log2', n_jobs=-1)
best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Stacking Ensemble
base_models = [
    ('rf', best_rf),
    ('gb', gb_model),
    ('svm', SVC(probability=True, random_state=42))  # SVC needs probability=True for stacking
]
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_test)

# Evaluation
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("AUC-ROC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]))

print("\nGradient Boosting:")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))
print("AUC-ROC:", roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1]))

print("\nStacking Ensemble:")
print("Accuracy:", accuracy_score(y_test, stacking_pred))
print(classification_report(y_test, stacking_pred))
print("AUC-ROC:", roc_auc_score(y_test, stacking_model.predict_proba(X_test)[:, 1]))

# Example of error analysis.
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, stacking_pred)
print("\nConfusion Matrix (Stacking):")
print(conf_matrix)

# Example of looking at misclassified examples.
misclassified = X_test[y_test != stacking_pred]
misclassified_y = y_test[y_test != stacking_pred]
print("\nNumber of misclassified examples:", len(misclassified))
print("First 5 misclassified examples:")
print(misclassified.head())
print("First 5 misclassified y_values:")
print(misclassified_y.head())

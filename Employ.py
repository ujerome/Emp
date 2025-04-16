# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 20:15:21 2025

@author: uwany
"""

# -*- coding: utf-8 -*-
"""
XGBoost Multi-Class Classification with Enhanced Accuracy
Created on Apr 15, 2025
@author: uwany
"""

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

# ------------------ Load and Clean Dataset ------------------

df = pd.read_csv(r"C:\Users\uwany\Downloads\EDUCATION DATASET.csv")
df.rename(columns={"Educaional_level": "Educational_level"}, inplace=True)
df_clean = df.dropna(subset=["LFP", "Educational_level", "Field_of_education", "TVT2"])

# Feature Engineering: Age Grouping
df_clean["Age_group"] = pd.cut(df_clean["Age"], bins=[15, 25, 35, 50, 65, 100], labels=["15-25", "26-35", "36-50", "51-65", "65+"])

features = ["Educational_level", "Field_of_education", "TVT2", "Sex", "Age_group"]
target = "LFP"
df_model = df_clean[features + [target]]

X = df_model.drop(columns=target)
y = df_model[target]

# ------------------ Encode Target ------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ------------------ Split Data ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ------------------ Preprocessing ------------------

categorical_features = ["Educational_level", "Field_of_education", "TVT2", "Sex", "Age_group"]
numerical_features = []

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ------------------ Pipeline with XGBoost ------------------

smote = SMOTENC(categorical_features=[0, 1, 2, 3, 4], random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    ))
])

# ------------------ Hyperparameter Tuning ------------------

param_dist = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__gamma': [0, 1, 5],
    'classifier__reg_alpha': [0, 0.5, 1],
    'classifier__reg_lambda': [1, 1.5, 2]
}

# Using 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=4,
    random_state=42,
    verbose=1
)

# ------------------ Train Model ------------------

random_search.fit(X_train, y_train)

# ------------------ Evaluation ------------------

y_pred = random_search.predict(X_test)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("🔍 Best Parameters Found:")
print(random_search.best_params_)

print("\n📊 Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# ------------------ Plot Confusion Matrix ------------------

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test_labels,
    y_pred_labels,
    display_labels=label_encoder.classes_,
    cmap='Blues',
    ax=ax
)
plt.title("📊 Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# ------------------ Save Model and Encoder ------------------

os.makedirs("saved_models", exist_ok=True)
joblib.dump(random_search.best_estimator_, "saved_models/best_xgboost_pipeline_optimized.pkl")
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")

print("\n✅ Model and label encoder saved successfully!")

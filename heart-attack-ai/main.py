import pandas as pd
import numpy as np
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("Please install imbalanced-learn package by running: python -m pip install imbalanced-learn")

# Wczytaj dane
df = pd.read_csv("data.csv", na_values=['?'])

# Usuń białe znaki z nazw kolumn (usuwanie spacji na końcu i początku)
df.columns = df.columns.str.strip()

# Sprawdź, czy 'num' to binary target; jeśli nie, ustaw: "y = (df['num'] > 0).astype(int)" dla binaryfikacji
if df['num'].nunique() > 2:
    df['num'] = (df['num'] > 0).astype(int)

# Detekcja feature’ów numerycznych i kategorycznych
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove('num')
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipeline do preprocessingu
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Podział na X/y
X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Zbalansowanie klasy 1 przez SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

# Model RandomForest z tuningiem
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print('Best Parameters:', grid.best_params_)

# Predykcja i ewaluacja
preds = grid.predict(X_test_prep)
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, preds))
auc = roc_auc_score(y_test, grid.predict_proba(X_test_prep)[:,1])
print(f'ROC AUC: {auc:.4f}')

# Wykres ważności cech
X_train_transformed = preprocessor.fit_transform(X_train)

if cat_features:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_features)
    feature_names = np.concatenate([num_features, cat_feature_names])
else:
    feature_names = np.array(num_features)

# Oblicz feature importance
importances = grid.best_estimator_.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Wykres
plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

import joblib
joblib.dump(grid.best_estimator_, "rf_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Zapisano modele!")
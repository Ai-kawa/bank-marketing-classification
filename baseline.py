import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Загрузка данных
# ============================================================

data = pd.read_csv("data/bank-full.csv", sep=";")

# Убираем "duration" — известен только после звонка (data leakage)
data = data.drop(columns=["duration"])


# ============================================================
# 2. Признаки и целевая переменная
# ============================================================

X = data.drop("y", axis=1)
y = data["y"].map({"no": 0, "yes": 1})

categorical_features = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]


# ============================================================
# 3. Разделение данных
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
)


# ============================================================
# 4. Препроцессинг
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", StandardScaler(), numeric_features)
    ]
)


# ============================================================
# 5. Logistic Regression (с балансировкой классов)
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])


# ============================================================
# 6. Обучение
# ============================================================

print("=" * 60)
print("ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ (Logistic Regression)")
print("=" * 60)

model.fit(X_train, y_train)


# ============================================================
# 7. Предсказание
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. Метрики
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)


# ============================================================
# 9. Результаты
# ============================================================

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ LOGISTIC REGRESSION")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

class_names = ["no", "yes"]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(confusion_matrix(y_test, y_pred))

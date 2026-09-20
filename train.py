import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. Загрузка данных
# ============================================================
# Файл использует ';' как разделитель и кавычки вокруг строк —
# особенность исходного датасета UCI Bank Marketing.

data = pd.read_csv("data/bank-full.csv", sep=";")

print("=" * 60)
print("АНАЛИЗ ДАТАСЕТА")
print("=" * 60)

print(f"Количество объектов: {len(data)}")
print(f"Количество признаков (исходно): {data.shape[1] - 1}")

print("\nПропуски:")
print(data.isnull().sum())

print("\nРаспределение целевой переменной:")
print(data["y"].value_counts())
print(data["y"].value_counts(normalize=True))


# ============================================================
# 2. Удаление признака с утечкой данных (data leakage)
# ============================================================
# Согласно документации датасета, признак "duration" (длительность
# последнего звонка) известен только ПОСЛЕ завершения звонка.
# Использовать его для реалистичного прогноза "до звонка" нельзя —
# это классический случай утечки целевой информации в признаки.
# Поэтому мы исключаем его из набора признаков.

data = data.drop(columns=["duration"])


# ============================================================
# 3. График распределения классов
# ============================================================

plt.figure(figsize=(6, 5))
data["y"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Распределение объектов по классам (y)")
plt.xlabel("Класс (подписка на депозит)")
plt.ylabel("Количество объектов")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.close()


# ============================================================
# 4. Признаки и целевая переменная
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
# 5. Разделение данных (train / val / test)
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
)

print("\n" + "=" * 60)
print("РАЗМЕРЫ ВЫБОРОК")
print("=" * 60)
print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")


# ============================================================
# 6. Препроцессинг: One-Hot для категорий + масштабирование чисел
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("numeric", StandardScaler(), numeric_features)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)
X_test_encoded = preprocessor.transform(X_test)

print("\nРазмерность после препроцессинга:")
print(f"Train: {X_train_encoded.shape}")
print(f"Validation: {X_val_encoded.shape}")
print(f"Test: {X_test_encoded.shape}")


# ============================================================
# 7. Веса классов (компенсация дисбаланса ~88% / ~12%)
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}

print("\nВеса классов для компенсации дисбаланса:")
print(class_weight)


# ============================================================
# 8. Архитектура нейронной сети
# ============================================================

input_size = X_train_encoded.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_size,)),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

print("\n" + "=" * 60)
print("АРХИТЕКТУРА")
print("=" * 60)
model.summary()


# ============================================================
# 9. Early Stopping
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


# ============================================================
# 10. Обучение
# ============================================================

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ")
print("=" * 60)

history = model.fit(
    X_train_encoded, y_train,
    validation_data=(X_val_encoded, y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\nКоличество фактически выполненных эпох: {len(history.history['loss'])}")


# ============================================================
# 11. График Loss
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Изменение функции потерь")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_history.png", dpi=300)
plt.close()


# ============================================================
# 12. График Accuracy
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Изменение точности модели")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_history.png", dpi=300)
plt.close()


# ============================================================
# 13. Оценка модели на тесте
# ============================================================

test_results = model.evaluate(X_test_encoded, y_test, verbose=0)
test_loss, test_accuracy, test_auc = test_results

y_probability = model.predict(X_test_encoded, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)


# ============================================================
# 14. Метрики
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-score:      {f1:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"ROC-AUC:       {roc_auc:.4f}")


# ============================================================
# 15. Classification Report
# ============================================================

class_names = ["no", "yes"]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))


# ============================================================
# 16. Матрица ошибок
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="rocket_r")
plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()


# ============================================================
# 17. ROC-кривая
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_probability)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайный классификатор")
plt.title("ROC-кривая")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()


# ============================================================
# 18. Сохранение модели и препроцессора
# ============================================================

model.save("model/bank_marketing_model.keras")
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("\nМодель сохранена:      model/bank_marketing_model.keras")
print("Препроцессор сохранён: model/preprocessor.pkl")

print("\nГрафики сохранены:")
print("class_distribution.png")
print("loss_history.png")
print("accuracy_history.png")
print("confusion_matrix.png")
print("roc_curve.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Параметры эксперимента

N_RUNS = 5
RANDOM_STATES = [0, 1, 2, 3, 4]

categorical_features = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]


def build_model(input_size):
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
        metrics=["accuracy"]
    )

    return model


def run_single_experiment(data, random_state):
    X = data.drop("y", axis=1)
    y = data["y"].map({"no": 0, "yes": 1})

    # Разделение данных

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.20,
        random_state=random_state,
        stratify=y_temp
    )

    # Подготовка признаков

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_val_encoded = preprocessor.transform(X_val)
    X_test_encoded = preprocessor.transform(X_test)

    # Балансировка классов

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    class_weight = {
        0: class_weights_array[0],
        1: class_weights_array[1]
    }

    model = build_model(X_train_encoded.shape[1])

    # Остановка при отсутствии улучшения

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    model.fit(
        X_train_encoded,
        y_train,
        validation_data=(X_val_encoded, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0
    )

    # Предсказание

    y_probability = model.predict(
        X_test_encoded,
        verbose=0
    ).ravel()

    y_pred = (y_probability >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_probability
        )
    }


# Запуск экспериментов

data = pd.read_csv("data/bank-full.csv", sep=";")
data = data.drop(columns=["duration"])

results = []

print("=" * 60)
print(f"СТАТИСТИЧЕСКАЯ ОЦЕНКА МОДЕЛИ ({N_RUNS} ЗАПУСКОВ)")
print("=" * 60)

for i, rs in enumerate(RANDOM_STATES, start=1):
    print(f"\nЗапуск {i}/{N_RUNS} (random_state={rs})...")

    metrics = run_single_experiment(data, rs)
    results.append(metrics)

    print(
        f"  Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )


# Сводная таблица

results_df = pd.DataFrame(results)
results_df.index = [
    f"Запуск {i + 1}"
    for i in range(N_RUNS)
]

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА ПО ВСЕМ ЗАПУСКАМ")
print("=" * 60)

print(results_df.round(4))

summary = results_df.agg(["mean", "std"])


# Средние значения и стандартное отклонение

print("\n" + "=" * 60)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 60)

for column in results_df.columns:
    mean_value = results_df[column].mean()
    std_value = results_df[column].std()

    print(
        f"{column:12s}: "
        f"{mean_value:.4f} ± {std_value:.4f}"
    )


results_df.to_csv("statistical_evaluation_results.csv")

print(
    "\nРезультаты сохранены: "
    "statistical_evaluation_results.csv"
)


# График результатов

fig, ax = plt.subplots(figsize=(9, 5))

means = results_df.mean()
stds = results_df.std()

bars = ax.bar(
    means.index,
    means.values,
    yerr=stds.values,
    capsize=6,
    color="#4C72B0"
)

ax.set_ylabel("Значение метрики")
ax.set_title(
    f"Статистическая устойчивость метрик "
    f"({N_RUNS} запусков)"
)
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)

for bar, mean_value in zip(bars, means.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        mean_value + 0.02,
        f"{mean_value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(
    "statistical_evaluation.png",
    dpi=300
)
plt.close()

print("График сохранён: statistical_evaluation.png")
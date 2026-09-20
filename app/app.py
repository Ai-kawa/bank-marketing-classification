from flask import Flask, render_template, request
import pandas as pd
import joblib
from tensorflow import keras


# Загружаем модель и обработчик данных
model = keras.models.load_model("model/bank_marketing_model.keras")
preprocessor = joblib.load("model/preprocessor.pkl")

app = Flask(__name__)

FEATURES = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month",
    "campaign", "pdays", "previous", "poutcome"
]

NUMERIC_FEATURES = {
    "age", "balance", "day", "campaign", "pdays", "previous"
}


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    probability = None

    if request.method == "POST":

        row = {}

        # Получаем данные из формы
        for feature in FEATURES:
            value = request.form[feature]

            if feature in NUMERIC_FEATURES:
                row[feature] = float(value)
            else:
                row[feature] = value

        client_data = pd.DataFrame([row])

        # Подготавливаем данные так же, как при обучении
        client_encoded = preprocessor.transform(client_data)

        # Получаем вероятность положительного класса
        proba_yes = float(
            model.predict(client_encoded, verbose=0)[0][0]
        )

        if proba_yes >= 0.5:
            result = "Да, вероятно откроет вклад"
        else:
            result = "Нет, вероятно откажется"

        probability = round(proba_yes * 100, 2)

    return render_template(
        "index.html",
        result=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
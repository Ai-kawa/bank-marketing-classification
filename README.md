# Bank Marketing — прогноз подписки на срочный вклад

Нейросетевое приложение для анализа данных: бинарная классификация клиентов
банка по вероятности подписки на срочный вклад по итогам телефонного звонка.
Датасет: [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Структура проекта

```
bank_marketing_project/
├── data/
│   └── bank-full.csv          # исходный датасет (45211 строк)
├── model/                     # сюда сохраняются обученные артефакты
│   ├── bank_marketing_model.keras
│   └── preprocessor.pkl
├── app/
│   ├── app.py                 # Flask-приложение для инференса
│   └── templates/
│       └── index.html
├── train.py                   # обучение нейросети (Keras)
├── baseline.py                 # baseline-модель (Logistic Regression)
├── requirements.txt
├── Dockerfile
└── compose.yaml
```

## 1. Установка зависимостей (локально)

```bash
pip install -r requirements.txt
```

## 2. Обучение модели

```bash
python train.py
```

После выполнения появятся:
- `model/bank_marketing_model.keras` и `model/preprocessor.pkl` — обученная модель и препроцессор;
- `class_distribution.png`, `loss_history.png`, `accuracy_history.png`, `confusion_matrix.png`, `roc_curve.png` — графики для пояснительной записки;
- в консоли — все метрики (accuracy, precision, recall, F1, macro F1, ROC-AUC) и classification report.

Для сравнения с baseline-моделью:

```bash
python baseline.py
```

## 3. Запуск веб-приложения локально (без Docker)

```bash
cd app
python app.py
```

Открыть в браузере: http://localhost:5000

## 4. Сборка и запуск в Docker

**Важно:** модель НЕ встраивается в образ — она подключается через volume
из директории `./model`, как того требует методичка.

```bash
docker build -t <ваш_dockerhub_логин>/bank-marketing:latest .
docker compose up --build
```

Открыть в браузере: http://localhost:5000

## 5. Публикация образа в DockerHub

```bash
docker login
docker push <ваш_dockerhub_логин>/bank-marketing:latest
```

## Примечание по признакам

Признак `duration` (длительность звонка) исключён из обучения, так как
он становится известен только после завершения звонка — использование
этого признака привело бы к утечке данных (data leakage) и нереалистично
завышенным метрикам.

import numpy as np
from tensorflow.keras.models import load_model
import os

# Названия классов в том порядке, в котором они обычно представлены в sklearn
CLASS_NAMES = ['Setosa', 'Versicolour', 'Virginica']

# Путь к модели относительно расположения файла
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris_model.keras')
model = None


def load_iris_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    return model


def predict_iris(features: list) -> str:
    """
    features: список из 4 чисел с плавающей точкой [sepal_length, sepal_width, petal_length, petal_width]
    Возвращает: название предсказанного класса
    """
    model = load_iris_model()
    # Преобразование в numpy array и добавление размерности batch
    input_data = np.array([features], dtype=np.float32)

    # Предсказание
    preds = model.predict(input_data, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]

    return CLASS_NAMES[class_idx]
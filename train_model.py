"""
Скрипт обучения модели классификации ирисов Фишера.
Сохраняет модель в iris_model.keras в формате, совместимом с model_utils.py.
Запуск: python train_model.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# 1. Данные
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target
y_cat = to_categorical(y, num_classes=3)

# ВАЖНО: обучение БЕЗ нормализации, чтобы форма могла передавать
# сырые сантиметры (5.1, 3.5, 1.4, 0.2) напрямую.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 2. Модель
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Обучение
model.fit(X_train, y_train, epochs=150, batch_size=8,
          validation_split=0.1, verbose=0)

# 4. Проверка
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность на тестовой выборке: {acc:.4f}")

# 5. Сохранение
model.save('iris_model.keras')
print("✅ Модель сохранена: iris_model.keras")

# 6. Быстрая самопроверка на эталонном примере Setosa
sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
pred = model.predict(sample, verbose=0)
print("Тест [5.1, 3.5, 1.4, 0.2] →", ['Setosa','Versicolour','Virginica'][int(np.argmax(pred))])
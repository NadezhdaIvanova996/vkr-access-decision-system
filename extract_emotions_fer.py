import numpy as np
import random
from datetime import datetime

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def extract_emotions(frame):
    """
    Имитация модуля распознавания эмоций из Проектной темы 1.
    
    На текущий момент используется синтетическая генерация вероятностей эмоций,
    так как реальный модуль от другого студента ещё не передан.
    Позже будет заменён на настоящую модель.
    
    Возвращает массив вероятностей 7 эмоций (сумма = 1.0).
    """
    # Для воспроизводимости + разнообразия
    random.seed(int(datetime.now().timestamp() * 1000) % 100000)

    # Случайно выбираем "доминирующую" эмоцию с высокой вероятностью
    dominant_idx = random.randint(0, 6)
    
    # Базовое распределение с акцентом на доминирующую эмоцию
    base = np.zeros(7)
    base[dominant_idx] = 0.70  # сильный акцент на одну эмоцию

    # Добавляем шум на остальные эмоции
    noise = np.random.dirichlet(np.ones(7) * 1.2)
    emotion_probs = base + noise * 0.30
    
    # Ограничиваем и нормализуем
    emotion_probs = np.clip(emotion_probs, 0.01, 1.0)
    emotion_probs = emotion_probs / emotion_probs.sum()

    return emotion_probs
# train_risk_model.py — обучение модели риска допуска посетителей
# с 12 признаками: 11 базовых + suspicious_score от Темы 2

import os
import cv2
import face_recognition
import numpy as np
import pickle
import xgboost as xgb
import random
from datetime import datetime
from tqdm import tqdm
# ====================== НАСТРОЙКИ ======================
FER_TRAIN_DIR = "data/train"
KNOWN_FACES_DIR = "known_faces"
MODEL_PATH = "models/risk_model.pkl"

# Порядок эмоций в FER2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NEGATIVE_EMOTIONS_IDX = [0, 2, 5]   # angry, fear, sad

# Для воспроизводимости
random.seed(42)
np.random.seed(42)

# =======================================================

def extract_features(img_path, known_encodings, known_names):
    """
    Извлекает 12 признаков + генерирует метку риска
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    if not encodings:
        return None, None

    enc = encodings[0]
    face_distances = face_recognition.face_distance(known_encodings, enc)
    min_dist = min(face_distances) if len(face_distances) > 0 else 1.0
    is_known = 1 if min_dist < 0.55 else 0

    # Эмоция из папки
    emotion = os.path.basename(os.path.dirname(img_path))
    emotion_idx = EMOTIONS.index(emotion) if emotion in EMOTIONS else 4

    emotion_probs = np.zeros(len(EMOTIONS))
    emotion_probs[emotion_idx] = 1.0

    # Контекстные признаки
    hour = datetime.now().hour / 24.0
    weekday = 1 if datetime.now().weekday() >= 5 else 0

    # ====================== 12-й ПРИЗНАК ОТ ТЕМЫ 2 ======================
    # suspicious_score — оценка опасности личных вещей (0.0 = чисто, 0.55 = подозрительно)
    suspicious_score = random.uniform(0.0, 0.55)

    # Собираем вектор из 12 признаков
    features = np.concatenate([
        [min_dist, is_known, hour, weekday, suspicious_score],   # 5 признаков
        emotion_probs                                            # 7 признаков
    ])

    # ====================== ГЕНЕРАЦИЯ МЕТКИ РИСКА ======================
    if is_known == 1:           # Известное лицо
        label = 1 if (emotion_idx in NEGATIVE_EMOTIONS_IDX and random.random() < 0.45) else 0
    else:                       # Неизвестное лицо
        if emotion_idx in NEGATIVE_EMOTIONS_IDX:
            label = 1
        elif emotion_idx == 4:   # neutral
            label = 1 if random.random() < 0.25 else 0
        else:
            label = 0

    return features, label


# ====================== ОСНОВНОЙ ПРОЦЕСС ======================

print("1. Загрузка известных лиц из папки 'known_faces/'...")
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(file.split('.')[0])

print(f"   Загружено {len(known_encodings)} известных лиц\n")

# ====================== СБОР ДАННЫХ ======================
print("2. Сбор всех изображений из FER2013 и случайная выборка 1000 фото...")

all_images = []
for emotion in EMOTIONS:
    emotion_dir = os.path.join(FER_TRAIN_DIR, emotion)
    if not os.path.exists(emotion_dir):
        print(f"   Папка {emotion} не найдена, пропускаем")
        continue

    for img_file in os.listdir(emotion_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_images.append(os.path.join(emotion_dir, img_file))

print(f"   Всего найдено фотографий: {len(all_images)}")

random.shuffle(all_images)
selected_images = all_images[:1000]

X, y = [], []
print("3. Обработка выбранных 1000 изображений...")

for img_path in tqdm(selected_images, desc="Обработка изображений"):
    features, label = extract_features(img_path, known_encodings, known_names)
    if features is not None:
        X.append(features)
        y.append(label)

if not X:
    print("Ошибка: не удалось извлечь признаки!")
    exit()

X = np.array(X)
y = np.array(y)

print(f"\n   Собрано {len(X)} примеров для обучения")
print(f"   Количество признаков: {X.shape[1]}")   

# Баланс классов
unique, counts = np.unique(y, return_counts=True)
print("   Распределение меток риска:", dict(zip(unique, counts)))

# ====================== ОБУЧЕНИЕ МОДЕЛИ ======================
print("\n4. Обучение модели XGBoost...")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X, y)

# Сохранение
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Готово! Модель успешно обучена и сохранена в {MODEL_PATH}")
print(f"Количество признаков в модели: {model.n_features_in_}")
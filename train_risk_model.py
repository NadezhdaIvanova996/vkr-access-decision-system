# train_risk_model.py — обучение модели риска на FER2013 + face_recognition
# Случайная выборка 1000 фото
import os
import cv2
import face_recognition
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime
from tqdm import tqdm
import random  # для случайной выборки

# Пути
FER_TRAIN_DIR = "data/train"
KNOWN_FACES_DIR = "known_faces"
MODEL_PATH = "models/risk_model.pkl"

# Порядок эмоций FER2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(img_path, known_encodings, known_names):
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

    hour = datetime.now().hour / 24.0
    weekday = 1 if datetime.now().weekday() >= 5 else 0

    features = np.concatenate([
        [min_dist, is_known, hour, weekday],
        emotion_probs
    ])

    # Метка риска — сбалансированная
    if is_known == 0:  # Unknown
        label = 1 if emotion_idx in [0, 2, 5] else 0  # angry, fear, sad → 1
    else:
        label = 1 if emotion_idx in [0, 2, 5] else 0

    return features, label

# ────────────────────────────────────────────────────────────────
# Основной процесс
# ────────────────────────────────────────────────────────────────
print("1. Загрузка известных лиц...")
known_encodings, known_names = [], []
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(file.split('.')[0])

print(f"   Загружено {len(known_encodings)} известных лиц")

# ────────────────────────────────────────────────────────────────
# Сбор данных — СЛУЧАЙНАЯ ВЫБОРКА 1000 фото
# ────────────────────────────────────────────────────────────────
print("2. Сбор всех фото из FER2013 и случайная выборка 1000...")

all_images = []
for emotion in EMOTIONS:
    emotion_dir = os.path.join(FER_TRAIN_DIR, emotion)
    if not os.path.exists(emotion_dir):
        print(f"   Папка {emotion} не найдена, пропускаем")
        continue
    for img_file in os.listdir(emotion_dir):
        all_images.append(os.path.join(emotion_dir, img_file))

print(f"   Всего найдено фото: {len(all_images)}")

# Перемешиваем
random.shuffle(all_images)

# Берём первые 1000
max_images = 1000
selected_images = all_images[:max_images]

X, y = [], []
print("3. Обработка выбранных 1000 фото...")
for img_path in tqdm(selected_images):
    features, label = extract_features(img_path, known_encodings, known_names)
    if features is not None:
        X.append(features)
        y.append(label)

if not X:
    print("Ошибка: не удалось извлечь признаки. Проверьте фото в data/train.")
    exit()

X = np.array(X)
y = np.array(y)
print(f"   Собрано {len(X)} примеров для обучения")

# Проверка баланса классов
unique, counts = np.unique(y, return_counts=True)
print("Распределение меток риска:", dict(zip(unique, counts)))

# ────────────────────────────────────────────────────────────────
# Обучение XGBoost
# ────────────────────────────────────────────────────────────────
print("4. Обучение модели...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)

# Сохраняем
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Готово! Модель сохранена в {MODEL_PATH}")
print(f"Количество признаков: {model.n_features_in_}")
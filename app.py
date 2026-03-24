import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
import shap
import os
import pandas as pd
from datetime import datetime
import time
import random

# =============================================
# Импорт модуля эмоций из Проектной темы 1
# =============================================
from extract_emotions_fer import extract_emotions, EMOTIONS

# =============================================
# Названия признаков (11 штук — соответствует обученной модели)
# =============================================
feature_names = [
    "Расстояние до известного лица",
    "Совпадение с известным (0/1)",
    "Час дня (0–1)",
    "Выходной день (0/1)",
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
]

# =============================================
# Настройка Streamlit приложения
# =============================================
st.set_page_config(page_title="Система допуска", layout="wide")
st.title("Система формирования решения о допуске посетителей")
st.markdown("ВКР Иванова Надежда Максимовна — Тема 3")

# =============================================
# Загрузка модели риска (XGBoost)
# =============================================
@st.cache_resource
def load_model():
    with open("models/risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()

# =============================================
# Загрузка известных лиц
# =============================================
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir("known_faces"):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = face_recognition.load_image_file(f"known_faces/{file}")
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(file.split('.')[0])
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# =============================================
# Основной интерфейс
# =============================================
col_video, col_result = st.columns([3, 2])

with col_video:
    st.subheader("Реальное видео")
    video_placeholder = st.empty()

with col_result:
    st.subheader("Результат")
    result_placeholder = st.empty()
    shap_placeholder = st.empty()

st.subheader("Таблица экспериментов")
table_placeholder = st.empty()

# =============================================
# Кнопки управления камерой
# =============================================
col1, col2, col3 = st.columns(3)
if col1.button("▶ Запустить камеру", type="primary"):
    st.session_state.camera_running = True
    st.session_state.camera_paused = False
if col2.button("⏸ Пауза"):
    st.session_state.camera_paused = True
if col3.button("▶ Возобновить"):
    st.session_state.camera_paused = False

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "camera_paused" not in st.session_state:
    st.session_state.camera_paused = False

if "experiment_table" not in st.session_state:
    st.session_state.experiment_table = pd.DataFrame(columns=["Время", "Имя", "Расстояние", "Риск %", "Решение"])

# =============================================
# Основной цикл обработки видео
# =============================================
if st.session_state.camera_running and not st.session_state.camera_paused:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Не удалось открыть камеру")
    else:
        ret, frame = cap.read()
        
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            name = "Unknown"
            risk_prob = 0.82
            min_dist = 1.0

            if encodings:
                enc = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
                face_distances = face_recognition.face_distance(known_encodings, enc)

                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    min_dist = round(float(face_distances[idx]), 3)

                # Создание вектора признаков (11 признаков)
                features = np.zeros((1, model.n_features_in_))
                features[0, 0] = min_dist
                features[0, 1] = 1.0 if name != "Unknown" else 0.0
                features[0, 2] = datetime.now().hour / 24.0

                # Получение эмоций из модуля Темы 1
                emotion_probs = extract_emotions(frame)
                features[0, 4:11] = emotion_probs

                # Синтетический признак от Темы 2 (анализ личных вещей)
                suspicious_score = random.uniform(0.0, 0.55)

                # Предсказание риска
                base_risk = model.predict_proba(features)[0][1]
                risk_prob = min(0.99, base_risk + suspicious_score * 0.28)

                # Получение SHAP значений
                shap_values = explainer.shap_values(features)[0]

            # Логика принятия решения
            if name == "Unknown":
                decision = "CHECK" if risk_prob < 0.50 else "DENY"
                color = (255, 165, 0) if decision == "CHECK" else (0, 0, 255)
            else:
                if risk_prob < 0.42:          # Порог для допуска известного лица
                    decision = "ALLOW"
                    color = (0, 255, 0)
                elif risk_prob < 0.62:
                    decision = "CHECK"
                    color = (255, 165, 0)
                else:
                    decision = "DENY"
                    color = (0, 0, 255)

            # Отрисовка на видео
            for (top, right, bottom, left) in locations:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.putText(frame, f"{name} | {decision} ({risk_prob:.1%})", (left, top-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                if min_dist < 1.0:
                    cv2.putText(frame, f"dist: {min_dist:.2f}", (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                # Отображение эмоции над лицом
                if len(features) >= 11:
                    main_emotion = EMOTIONS[np.argmax(features[0][4:11])]
                    cv2.putText(frame, f"Эмоция: {main_emotion}", (left, top-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            video_placeholder.image(frame, channels="BGR")

            # Вывод результата
            if decision == "ALLOW":
                result_placeholder.success(f"✅ ДОПУСТИТЬ — Уверенность {100 - int(risk_prob*100)}%")
            elif decision == "CHECK":
                result_placeholder.warning(f"⚠️ ПРОВЕРИТЬ — Риск {int(risk_prob*100)}%")
            else:
                result_placeholder.error(f"⛔ ОТКАЗАТЬ — Риск {int(risk_prob*100)}%")

            # SHAP объяснение
            try:
                expected = explainer.expected_value
                if isinstance(expected, (list, np.ndarray)):
                    expected = expected[1] if len(expected) > 1 else expected[0]
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=expected,
                    feature_names=feature_names[:len(shap_values)]
                )
                shap_placeholder.pyplot(shap.waterfall_plot(explanation))
            except Exception as e:
                shap_placeholder.text(f"SHAP не построен: {str(e)}")

            # Запись в таблицу экспериментов
            decision_rus = "ДОПУСТИТЬ" if decision == "ALLOW" else "ПРОВЕРИТЬ" if decision == "CHECK" else "ОТКАЗАТЬ"
            new_row = pd.DataFrame([{
                "Время": datetime.now().strftime("%H:%M:%S"),
                "Имя": name,
                "Расстояние": round(min_dist, 3) if min_dist < 1 else "—",
                "Риск %": round(risk_prob * 100, 1),
                "Решение": decision_rus
            }])
            st.session_state.experiment_table = pd.concat([st.session_state.experiment_table, new_row], ignore_index=True)
            table_placeholder.dataframe(st.session_state.experiment_table.tail(15), use_container_width=True)

        cap.release()
        time.sleep(8.0)
        st.rerun()

else:
    video_placeholder.info("Камера остановлена или на паузе. Нажмите «Запустить камеру».")
    table_placeholder.dataframe(st.session_state.experiment_table.tail(15), use_container_width=True)
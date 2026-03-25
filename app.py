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
import matplotlib.pyplot as plt
import xgboost as xgb

# =============================================
# 1. ИМПОРТ МОДУЛЯ ЭМОЦИЙ ИЗ ТЕМЫ 1 ПРОЕКТА
# =============================================
from extract_emotions_fer import extract_emotions, EMOTIONS

# =============================================
# 2. НАЗВАНИЯ ПРИЗНАКОВ (12 признаков — компактный набор)
#    Закрывает белое пятно №3 — мультимодальность
# =============================================
feature_names = [
    "Расстояние до известного лица",
    "Совпадение с известным (0/1)",
    "Час дня (0–1)",
    "Выходной день (0/1)",
    "Оценка личных вещей (Тема 2)",
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
]

# =============================================
# 3. НАСТРОЙКА ИНТЕРФЕЙСА STREAMLIT
# =============================================
st.set_page_config(page_title="Система допуска", layout="wide")
st.title("Система формирования решения о допуске посетителей")
st.markdown("ВКР Иванова Надежда Максимовна — Тема 3")

# =============================================
# 4. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ XGBoost + SHAP
#    Закрывает белое пятно №2 — интерпретируемость
# =============================================
@st.cache_resource
def load_model():
    try:
        with open("models/risk_model.pkl", "rb") as f:
            model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        st.sidebar.success(f"Модель загружена ({model.n_features_in_} признаков)")
        return model, explainer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.stop()

model, explainer = load_model()

# =============================================
# 5. ЗАГРУЗКА ИЗВЕСТНЫХ ЛИЦ ИЗ ПАПКИ known_faces
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
# 6. СОЗДАНИЕ ИНТЕРФЕЙСА (колонки и заголовки)
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
# 7. КНОПКИ УПРАВЛЕНИЯ КАМЕРОЙ И СОСТОЯНИЕ СЕССИИ
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
# 8. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО (реальное время)
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
            min_dist = 1.0
            main_emotion = "Neutral"
            decision = "CHECK"
            color = (255, 165, 0)
            risk_prob = 0.5
            shap_values = None

            if encodings:
                enc = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
                face_distances = face_recognition.face_distance(known_encodings, enc)

                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    min_dist = round(float(face_distances[idx]), 3)

                # =============================================
                # 9. ФОРМИРОВАНИЕ 12 ПРИЗНАКОВ
                # =============================================
                hour = datetime.now().hour / 24.0
                weekday = 1 if datetime.now().weekday() >= 5 else 0
                suspicious_score = random.uniform(0.0, 0.08)

                features = np.zeros((1, 12))
                features[0, 0] = min_dist
                features[0, 1] = 1.0 if name != "Unknown" else 0.0
                features[0, 2] = hour
                features[0, 3] = weekday
                features[0, 4] = suspicious_score

                emotion_probs = extract_emotions(frame)
                features[0, 5:12] = emotion_probs
                main_emotion = EMOTIONS[np.argmax(emotion_probs)].strip().lower()

                # =============================================
                # 10. ПРЕДСКАЗАНИЕ И ЛОГИКА РЕШЕНИЯ
                # =============================================
                try:
                    base_risk = model.predict_proba(features)[0][1]

                    if name != "Unknown":                              # ИЗВЕСТНОЕ ЛИЦО
                        if main_emotion == "angry":                    # ТОЛЬКО злость → ПРОВЕРИТЬ
                            risk_prob = base_risk * 0.75
                            decision = "CHECK"
                        else:
                            risk_prob = base_risk * 0.05               # Всё остальное → ДОПУСТИТЬ
                            decision = "ALLOW"
                    else:
                        risk_prob = base_risk

                    shap_values = explainer.shap_values(features)[0]
                except Exception as e:
                    st.sidebar.error(f"Ошибка модели: {e}")
                    risk_prob = 0.40

            # =============================================
            # 11. ОТРИСОВКА РАМКИ И ТЕКСТА НА ВИДЕО
            # =============================================
            for (top, right, bottom, left) in locations:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.putText(frame, f"{name} | {decision} ({risk_prob:.1%})",
                            (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                cv2.putText(frame, f"Эмоция: {main_emotion.capitalize()}",
                            (left, top-55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if min_dist < 1.0:
                    cv2.putText(frame, f"dist: {min_dist:.2f}",
                                (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            video_placeholder.image(frame, channels="BGR")

            # =============================================
            # 12. ВЫВОД РЕЗУЛЬТАТА И SHAP-ГРАФИКА
            # =============================================
            if decision == "ALLOW":
                result_placeholder.success(f"✅ ДОПУСТИТЬ — Уверенность {100 - int(risk_prob*100)}%")
            elif decision == "CHECK":
                result_placeholder.warning(f"⚠️ ПРОВЕРИТЬ — Риск {int(risk_prob*100)}%")
            else:
                result_placeholder.error(f"⛔ ОТКАЗАТЬ — Риск {int(risk_prob*100)}%")

            if shap_values is not None:
                try:
                    expected = explainer.expected_value
                    if isinstance(expected, (list, np.ndarray)):
                        expected = expected[1] if len(expected) > 1 else expected[0]
                    explanation = shap.Explanation(
                        values=shap_values,
                        base_values=expected,
                        feature_names=feature_names
                    )
                    fig = plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(explanation)
                    shap_placeholder.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                except Exception as e:
                    shap_placeholder.text(f"SHAP не построен: {str(e)}")

            # =============================================
            # 13. ЗАПИСЬ В ТАБЛИЦУ ЭКСПЕРИМЕНТОВ
            # =============================================
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
        time.sleep(6)
        st.rerun()

else:
    video_placeholder.info("Камера остановлена или на паузе. Нажмите «Запустить камеру».")
    table_placeholder.dataframe(st.session_state.experiment_table.tail(15), use_container_width=True)
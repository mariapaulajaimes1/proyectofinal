import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Encabezado y bienvenida
st.title("Feelify: Your Mood, Your Music 🎶")
st.subheader("Analizando tu estado de ánimo para ofrecerte la música perfecta")

# Cargar el modelo
model = load_model("streamlit_app/teachable_machine/emotion_model.h5")

# Mapear las emociones detectadas a nombres
emotion_labels = ["feliz", "triste", "enojado"]  # Asegúrate que coincidan con las clases de tu modelo

# Función para predecir emoción
def predict_emotion(frame):
    # Preprocesamiento de la imagen (tamaño y normalización)
    resized_frame = cv2.resize(frame, (224, 224))  # Tamaño requerido por el modelo
    normalized_frame = resized_frame / 255.0  # Normalizar píxeles entre 0 y 1
    # Predicción
    prediction = model.predict(np.expand_dims(normalized_frame, axis=0))
    emotion_index = np.argmax(prediction)  # Obtener el índice de la emoción más probable
    emotion = emotion_labels[emotion_index]  # Obtener la emoción
    return emotion

# Configuración de la cámara
def capture_frame():
    # Abre la cámara y captura un solo fotograma
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()
    if ret:
        return frame
    else:
        return None

# Mostrar video en tiempo real
frame = capture_frame()
if frame is not None:
    st.image(frame, channels="BGR", caption="Captura de la cámara en vivo")

# Predecir estado de ánimo y mostrarlo
if frame is not None:
    emotion = predict_emotion(frame)  # Analizar la emoción de la imagen
    st.write(f"Estado de ánimo detectado: {emotion}")

    # Crear playlists de ejemplo para cada estado de ánimo
    playlists = {
        "feliz": ["streamlit_app/assets/Canción_1.mp3", "streamlit_app/assets/Canción_2.mp3"],
        "triste": ["streamlit_app/assets/Canción_3.mp3", "streamlit_app/assets/Canción_4.mp3"],
        "enojado": ["streamlit_app/assets/Canción_5.mp3", "streamlit_app/assets/Canción_6.mp3"]
    }

    # Función para recomendar playlist
    def recommend_playlist(emotion):
        return playlists.get(emotion, ["streamlit_app/assets/Canción_genérica.mp3"])

    # Mostrar playlist en la app, solo si la emoción ha sido detectada
    playlist = recommend_playlist(emotion)
    st.write("Tu playlist personalizada:")
    for song in playlist:
        st.audio(song, format="audio/mp3")
else:
    st.error("No se pudo acceder a la cámara.")








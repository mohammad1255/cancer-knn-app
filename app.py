# save this file as app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# === تحميل KNN و Scaler المحفوظين ===
scaler = joblib.load("model/scaler.pkl")
knn_model = joblib.load("model/best_knn_model.pkl")

# === إعادة بناء EfficientNet كـ feature extractor ===
base_model = EfficientNetB0(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# === معالجة الصورة واستخلاص السمات ===
def preprocess_and_extract_feature(image, feature_extractor, scaler):
    image = image.convert('RGB').resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    features = feature_extractor(img_tensor, training=False)
    features = GlobalAveragePooling2D()(features).numpy()
    return scaler.transform(features)

# === واجهة Streamlit ===
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")

st.title("🔬 Breast Cancer Classifier (EfficientNet + KNN)")
st.write("Upload a mammogram image and get a prediction.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # زر التنبؤ
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            features = preprocess_and_extract_feature(image, feature_extractor, scaler)
            prediction = knn_model.predict(features)[0]
            confidence = np.max(knn_model.predict_proba(features))

            label = "Malignant (سرطان خبيث)" if prediction == 1 else "Benign (ورم حميد)"
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence:.2%}")

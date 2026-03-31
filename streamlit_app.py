# =========================
# IMPORTS
# =========================
import PIL
import streamlit as st
from ultralytics import YOLO
import time
import cv2
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Microplastic Detection",
    page_icon="🔬",
    layout="wide"
)

# =========================
# LOAD CSS
# =========================
with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# REMOVE STREAMLIT FOOTER
# =========================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="title">NeerNetra</div>
<div class="subtitle">(Microplastic Detection System)</div>
<div class="subtitle">AI + Real-Time Monitoring</div>
""", unsafe_allow_html=True)

# =========================
# MODE SELECTOR
# =========================
mode = st.radio(
    "Select Mode",
    ["📤 Upload Detection", "🎥 Live Detection"],
    horizontal=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Controls")

    source_img = st.file_uploader(
        "Upload Image", type=("jpg", "jpeg", "png", "bmp", "webp"))

    confidence = float(st.slider(
        "Confidence", 25, 100, 40)) / 100

# =========================
# LOAD MODEL
# =========================
model_path = 'weights/t29.pt'

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error("Model loading failed")
    st.error(ex)

# =========================
# 📤 UPLOAD DETECTION
# =========================
if mode == "📤 Upload Detection":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Image Detection")

    if source_img:
        uploaded_image = PIL.Image.open(source_img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image, caption="Uploaded Image")

        if st.button("🔍 Detect Microplastics"):

            with st.spinner("Detecting microplastics..."):

                start_time = time.time()

                res = model.predict(uploaded_image, conf=confidence)

                end_time = time.time()

                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted, caption="Detected Image")

                count = len(res[0].boxes)

                st.markdown(
                    f"<p class='success-text'>Detected Particles: {count}</p>",
                    unsafe_allow_html=True
                )

                st.success(f"Prediction Time: {end_time - start_time:.2f} sec")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🎥 LIVE DETECTION
# =========================
if mode == "🎥 Live Detection":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Microplastic Detection")

    # 🔥 CHANGE THIS IF NEEDED (0 / 1 / 2)
    CAMERA_INDEX = 1  

    run = st.checkbox("Start Live Detection")

    FRAME_WINDOW = st.empty()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("❌ Camera not detected. Change CAMERA_INDEX.")
    else:
        st.success(f"✅ Using Camera Index: {CAMERA_INDEX}")

    if run and cap.isOpened():

        for _ in range(500):

            ret, frame = cap.read()

            if not ret:
                st.error("Camera error")
                break

            frame = cv2.resize(frame, (320, 240))

            # ===== DETECTION =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5,5), 0)

            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            contours, _ = cv2.findContours(
                opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            for contour in contours:
                if cv2.contourArea(contour) > 20:

                    x, y, w, h = cv2.boundingRect(contour)

                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)

                    cv2.putText(display, "Microplastic",
                                (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0,255,0), 1)

            FRAME_WINDOW.image(display, channels="BGR")

            time.sleep(0.5)  # stable detection

        cap.release()

    st.markdown('</div>', unsafe_allow_html=True)
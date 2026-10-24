
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Road DetectAI",
    page_icon="🛣️",
    layout="wide"
)

MODEL_PATH = "/content/drive/MyDrive/RoadDetectAI/training/pothole_model/weights/best.pt"

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111c30;
}
h1, h2, h3, p, label {
    color: #f1f5f9;
}
div.stButton > button {
    background: #ef4444;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #dc2626;
    color: white;
}
.metric-card {
    background: #162238;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263750;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

st.sidebar.title("🛣️ Road DetectAI")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Model Information", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered pothole detection")

if page == "Dashboard":
    st.title("Road Damage Detection")
    st.write("Upload a road image to detect potholes using YOLOv8.")

    if not Path(MODEL_PATH).exists():
        st.error("Model file not found. Please check the model path.")
        st.stop()

    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Run Road Damage Detection", use_container_width=True):
            with st.spinner("Detecting potholes..."):
                model = load_model()
                results = model.predict(
                    source=np.array(image),
                    conf=confidence,
                    verbose=False
                )

                result = results[0]
                annotated = result.plot()
                annotated = annotated[:, :, ::-1]

                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0

                image_area = image.width * image.height
                damage_area = 0
                rows = []

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [
                            round(float(v), 1)
                            for v in box.xyxy[0].tolist()
                        ]
                        conf = round(float(box.conf[0]), 3)
                        area = max(0, x2-x1) * max(0, y2-y1)
                        damage_area += area

                        rows.append({
                            "Class": "Pothole",
                            "Confidence": f"{conf:.1%}",
                            "X1": x1,
                            "Y1": y1,
                            "X2": x2,
                            "Y2": y2
                        })

                coverage = min(100, damage_area / image_area * 100) if image_area else 0
                severity = (
                    "High" if coverage >= 10 else
                    "Medium" if coverage >= 3 else
                    "Low" if count > 0 else
                    "No potholes detected"
                )

                with right:
                    st.subheader("Detected Potholes")
                    st.image(annotated, use_container_width=True)

                st.markdown("### Detection Summary")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f'<div class="metric-card"><p>Total Detections</p>'
                        f'<h2>{count}</h2></div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(
                        f'<div class="metric-card"><p>Bounding-box Coverage</p>'
                        f'<h2>{coverage:.2f}%</h2></div>',
                        unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(
                        f'<div class="metric-card"><p>Estimated Severity</p>'
                        f'<h2>{severity}</h2></div>',
                        unsafe_allow_html=True
                    )

                st.caption(
                    "Coverage is the summed bounding-box area divided by image "
                    "area; overlapping boxes may be counted more than once. "
                    "Severity is a simple demo estimate, not an engineering assessment."
                )

                st.markdown("### Detection Details")
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Detection Report (CSV)",
                        df.to_csv(index=False),
                        file_name="road_detectai_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("No potholes detected above the selected confidence threshold.")

else:
    if page == "Model Information":
        st.title("Model Information")
        st.write("**Model:** YOLOv8")
        st.write("**Detection class:** Pothole")
        st.write("**Confidence threshold:** Adjustable from the Dashboard.")
        st.write("**Inference:** Runs locally in the app environment.")
    else:
        st.title("About Road DetectAI")
        st.write(
            "Road DetectAI is a pothole detection demonstration designed "
            "to identify visible potholes in road images using computer vision."
        )
        st.info(
            "This is a prototype. Predictions may be incorrect and should "
            "not be used alone for road-safety or maintenance decisions."
        )

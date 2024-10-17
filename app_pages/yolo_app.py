import os
import uuid

import streamlit as st
from PIL import Image
from ultralytics import NAS, RTDETR, SAM, YOLO, YOLOWorld

YOLO_FILES_DIR = "yolo_files/"
os.makedirs(YOLO_FILES_DIR, exist_ok=True)

YOLO_MODELS = [
    os.path.join(YOLO_FILES_DIR, "yolo11.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov10n.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov9s.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov8n.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov7.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov6-n.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov5n.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov4n.pt"),
    os.path.join(YOLO_FILES_DIR, "yolov3n.pt"),
]

SAM_MODELS = [
    os.path.join(YOLO_FILES_DIR, "sam_b.pt"),
    os.path.join(YOLO_FILES_DIR, "sam2_b.pt"),
    os.path.join(YOLO_FILES_DIR, "mobile_sam.pt"),
]

NAS_MODELS = [
    os.path.join(YOLO_FILES_DIR, "yolo_nas_s.pt"),
]

RTDETR_MODELS = [
    os.path.join(YOLO_FILES_DIR, "rtdetr-1.pt"),
]

YOLOWORLD_MODELS = [
    os.path.join(YOLO_FILES_DIR, "yolov8s-worldv2.pt"),
]

if __name__ == "__page__":
    st.set_page_config(
        page_title="Object Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    run = None
    stop = None

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = uuid.uuid4()

    if "annotated_img" in st.session_state:
        st.session_state.annotated_img = None

    with st.sidebar:
        st.write("Object Detection Models")
        col1, col2 = st.columns(2)
        with col1:
            models_variation = st.selectbox(
                "Models", ["YOLO", "SAM", "NAS", "RTDETR", "YOLOWORLD"]
            )
            models_to_list = globals()[f"{models_variation}_MODELS"]
        with col2:
            model_chosen = st.selectbox("Variations", models_to_list)
        if models_variation == "YOLO":
            if "11" in model_chosen:
                with col1:
                    v11_variation = st.selectbox(
                        "Type",
                        [
                            "Object detection",
                            "Segmentation",
                            "Pose/Keypoints",
                            "Oriented detection",
                            "Classification",
                        ],
                    )
                with col2:
                    v11_size = st.selectbox(
                        "Size", ["XL", "Large", "Medium", "Small", "Nano"]
                    )

                base, ext = model_chosen.rsplit(".", 1)
                size_initial = v11_size.lower()[0]
                model_chosen = f"{base}{size_initial}.{ext}"

                if v11_variation != "Object detection":
                    base, ext = model_chosen.rsplit(".", 1)
                    if v11_variation == "Segmentation":
                        model_chosen = f"{base}-seg.{ext}"
                    elif v11_variation == "Pose/Keypoints":
                        model_chosen = f"{base}-pose.{ext}"
                    elif v11_variation == "Oriented detection":
                        model_chosen = f"{base}-obb.{ext}"
                    elif v11_variation == "Classification":
                        model_chosen = f"{base}-cls.{ext}"

                model = YOLO(model_chosen)
        elif models_variation == "SAM":
            model = SAM(model_chosen)
        elif models_variation == "NAS":
            model = NAS(model_chosen)
        elif models_variation == "RTDETR":
            model = RTDETR(model_chosen)
        elif models_variation == "YOLOWORLD":
            model = YOLOWorld(model_chosen)
        else:
            st.write("Model not found")

        run = st.button("Run", type="primary", use_container_width=True)
        stop = st.button("Stop", use_container_width=True, disabled=not run)

    uploaded_images = st.file_uploader(
        "Upload images",
        key=st.session_state.file_uploader_key,
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button(
        "Clear images",
        use_container_width=True,
        type="primary",
        disabled=len(uploaded_images) == 0,
    ):
        st.session_state.file_uploader_key = uuid.uuid4()
        st.rerun()

    col1, col2 = st.columns(2)
    for uploaded_image in uploaded_images:
        with col1:
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image")

        with col2:
            if run and image:
                results = model(image)
                st.session_state.annotated_img = Image.fromarray(
                    results[0].plot()[..., ::-1]
                )
                st.image(st.session_state.annotated_img, caption="Annotated Image")
    if stop:
        st.stop()

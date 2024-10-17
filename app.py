import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
    )

    multimodal_page = st.Page(
        page="app_pages/multimodal_app.py",
        title="Multimodal Chatbot",
        icon=":material/smart_toy:",
        default=True,
    )
    yolo_page = st.Page(
        page="app_pages/yolo_app.py",
        title="YOLO Object Detection",
        icon=":material/center_focus_strong:",
    )

    pages = [multimodal_page, yolo_page]
    pg = st.navigation(pages)
    pg.run()

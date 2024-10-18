import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
    )

    meeting_summary_page = st.Page(
        page="app_pages/meeting_summary_app.py",
        title="Meeting Summary",
        icon=":material/groups:",
        default=True,
    )
    news_chatbot_page = st.Page(
        page="app_pages/news_chatbot.py",
        title="News Chatbot",
        icon=":material/news:",
    )
    multimodal_page = st.Page(
        page="app_pages/multimodal_app.py",
        title="Multimodal Chatbot",
        icon=":material/smart_toy:",
    )
    yolo_page = st.Page(
        page="app_pages/yolo_app.py",
        title="YOLO Object Detection",
        icon=":material/center_focus_strong:",
    )

    pages = [meeting_summary_page, news_chatbot_page, multimodal_page, yolo_page]
    pg = st.navigation(pages)
    pg.run()

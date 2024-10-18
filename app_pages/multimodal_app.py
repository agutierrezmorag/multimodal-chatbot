import base64
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_MESSAGE = """Transcribe the data from this image into a structured format. \
Extract any text, numbers, or tables present, and return the data as a pandas DataFrame. \
Make sure the DataFrame is organized by columns for easy manipulation, with headers inferred  \
from the content where appropriate. Ensure the text is accurate and aligned with the layout in \
the image, and handle any irregularities like missing or misaligned data as best as possible. \
Do NOT tell the user how to extract the data, extract it yourself."""
HUMAN_MESSAGE = "Analyze the image and extract the data into a pandas DataFrame."

if __name__ == "__page__":
    if "response" not in st.session_state:
        st.session_state.response = None

    tracer = LangChainTracer(project_name="Multimodal Chatbot")

    st.title("Multimodal Chatbot")
    col1, col2 = st.columns(2)
    image_data = None
    api_key = None
    chat_model = None
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "API Key",
                type="password",
                value=openai_api_key or anthropic_api_key or "",
            )
        with col2:
            chosen_model = st.selectbox("Modelo IA", ["OpenAI", "Anthropic"])
            try:
                if chosen_model == "OpenAI":
                    chat_model = ChatOpenAI(
                        model="gpt-4o-mini", api_key=openai_api_key or api_key
                    )
                if chosen_model == "Anthropic":
                    chat_model = ChatAnthropic(
                        model="claude-3-haiku-20240307",
                        api_key=anthropic_api_key or api_key,
                    )
            except Exception as e:  # noqa: F841
                pass
        if not api_key:
            st.error("Por favor, ingrese su API Key")

        upload_method = st.selectbox("Metodo de Subida", ["URL", "Archivo"])
        if upload_method == "URL":
            image_url = st.text_input("URL de la Imagen")
            if image_url:
                st.image(image_url)
                image_data = base64.b64encode(httpx.get(image_url).content).decode(
                    "utf-8"
                )
        else:
            uploaded_image = st.file_uploader(
                "Subir Imagen", type=["jpg", "jpeg", "png"]
            )
            if uploaded_image:
                st.image(uploaded_image)
                image_data = base64.b64encode(uploaded_image.read()).decode("utf-8")

    col3, col4 = st.columns(2)
    with col3:
        system = st.text_area(
            label="System Message (Prompt)",
            value=SYSTEM_MESSAGE,
            help="El mensaje que recibe el modelo antes de la query, donde se le da contexto al modelo sobre el problema a resolver. Editable.",
        )
    with col4:
        human = st.text_area(
            label="Human Message (Query)",
            value=HUMAN_MESSAGE,
            help="El mensaje que recibe el modelo como input, donde se le pide resolver un problema. Editable.",
        )
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": human,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    system_message = SystemMessage(
        content=system,
    )
    try:
        if st.button(
            "Enviar query",
            disabled=(api_key is None and image_data is None),
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Esperando respuesta del modelo..."):
                st.session_state.response = chat_model.invoke(
                    [system_message, human_message], config={"callbacks": [tracer]}
                )
    except Exception as e:
        st.error(e)

    if st.session_state.response:
        st.write(st.session_state.response.content)

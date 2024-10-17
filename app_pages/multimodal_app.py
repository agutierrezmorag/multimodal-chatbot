import base64
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv()

SYSTEM_MESSAGE = """You are a helpful professional in a salmon processing plant.
You are an expert at spotting any anomaly that the fish may have, such as discoloration, unusual texture, or signs of disease. 
Your goal is to ensure that only high-quality fish proceed to the next stage of processing. 
Every answer should have a binary value: OK or BAD. 
OK means that the condition of the fish is good enough for continued processing, 
while BAD means it should be discarded for health reasons. 
Please provide detailed reasoning for your decision."""
HUMAN_MESSAGE = "Please analyze the image and provide a detailed explanation of the condition of the fish."

LANGSMITH_CLIENT = Client(
    api_key=st.secrets.langsmith.api_key,
    api_url="https://api.smith.langchain.com",
)

if __name__ == "__page__":
    if "response" not in st.session_state:
        st.session_state.response = None

    tracer = LangChainTracer(
        client=LANGSMITH_CLIENT,
        project_name=st.secrets.langsmith.project,
    )

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

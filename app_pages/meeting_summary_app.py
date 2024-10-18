import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

client = OpenAI()

sys_msg_template = """Realiza un resumen del siguiente contexto, siguiendo este formato:
# Resumen
Aqui escribiras el resumen del contexto.
# Puntos clave
Aqui escribiras los puntos clave del contexto.
# Observaciones
Aqui escribiras las observaciones del contexto.
# Conclusiones
Aqui escribiras las conclusiones del contexto.
        
El contexto es el siguiente:
{context}"""


@st.cache_data(show_spinner=False)
def transcribe_test_audio():
    """Transcribes a test audio file."""
    audio_file = open("Untitled notebook.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="text"
    )
    return transcription


if __name__ == "__page__":
    tracer = LangChainTracer(project_name="Meeting Summarizer")

    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    with st.sidebar:
        work_method = st.selectbox("Type", ["Audio", "Transcription"])
        if work_method == "Audio":
            st.audio("Untitled notebook.wav")
            st.caption("[source](https://www.youtube.com/watch?v=lBVtvOpU80Q)")
            if st.button("Transcribe", use_container_width=True):
                st.session_state.transcription = transcribe_test_audio()
        st.session_state.transcription = st.text_area(
            "Transcription", value=st.session_state.transcription
        )
        sys_msg = st.text_area("Prompt", value=sys_msg_template)

        generate_summary = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.transcription is None,
        )

    if generate_summary:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=50,
            length_function=len,
        )
        texts = text_splitter.split_text(st.session_state.transcription)
        docs = []
        for text in texts:
            docs.append(Document(page_content=text, metadata={"source": work_method}))

        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    sys_msg,
                )
            ]
        )
        chain = create_stuff_documents_chain(llm, prompt)

        st.session_state.summary = chain.invoke(
            {"context": docs}, config={"callbacks": [tracer]}
        )

    if st.session_state.summary:
        st.write(st.session_state.summary)

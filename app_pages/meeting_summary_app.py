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

sys_msg_template = """Genera un resumen detallado basado en el siguiente contexto. \
Organiza la información utilizando los siguientes encabezados:
# Resumen
Ofrece un resumen claro y conciso del contenido presentado en el contexto. Destaca los temas principales.

## Puntos clave
Enumera los puntos más importantes o destacables del contexto, incluyendo datos relevantes, decisiones tomadas, y aspectos clave.

## Observaciones
Incluye cualquier observación adicional que pueda ser relevante, como comentarios sobre la dinámica de la reunión, temas secundarios, o información complementaria.

## Conclusiones
Proporciona las conclusiones o implicaciones derivadas del contexto, resaltando posibles próximas acciones o decisiones a tomar.

El contexto es el siguiente:
{context}"""

DEFAULT_CONTEXT = """[Moderador]: Buenos días a todos, gracias por unirse a la reunión de hoy. Vamos a empezar con la actualización del proyecto de marketing. Carlos, ¿puedes comenzar?

[Carlos]: Claro, hemos visto un crecimiento del 15% en el tráfico de nuestra campaña en redes sociales durante la última semana. Sin embargo, las conversiones no han subido al mismo ritmo. Estamos planeando ajustar los anuncios para enfocarnos más en la conversión de clientes potenciales.

[Moderador]: Gracias, Carlos. ¿Hay alguna estimación de cuánto tiempo tomará ese ajuste?

[Carlos]: Esperamos tener los cambios implementados para el final de la semana.

[Moderador]: Perfecto. Pasando al siguiente punto, necesitamos revisar el presupuesto para el próximo trimestre. Sandra, ¿puedes actualizar al equipo sobre las proyecciones financieras?

[Sandra]: Sí, con base en los gastos actuales y las expectativas de ingresos, proyectamos un aumento del 10% en el presupuesto de marketing, pero habrá una reducción en las áreas de desarrollo de producto debido a un retraso en la contratación de personal clave.

[Moderador]: ¿Cuándo planeamos reanudar las contrataciones?

[Sandra]: Probablemente a partir del próximo mes, una vez que tengamos la aprobación final del equipo ejecutivo.

[Moderador]: Bien, entonces seguimos en ese plan. ¿Alguien tiene alguna otra cosa que añadir antes de cerrar la reunión?

[Carlos]: Solo recordar que tenemos la reunión con el equipo de ventas mañana a las 10 AM.

[Moderador]: Gracias, Carlos. Si no hay nada más, terminamos aquí. Que tengan un buen día."""


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
            "Example Transcription",
            value=st.session_state.get("transcription", DEFAULT_CONTEXT)
            or DEFAULT_CONTEXT,
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

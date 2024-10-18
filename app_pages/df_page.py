import pandas as pd
import streamlit as st
from langchain.callbacks.tracers import LangChainTracer
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


@st.cache_resource
def get_agent(df):
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        verbose=True,
        allow_dangerous_code=True,
    )
    return agent


if __name__ == "__page__":
    if "agent_response" not in st.session_state:
        st.session_state.agent_response = None

    tracer = LangChainTracer(project_name="DF Q&A")

    df = pd.read_csv("SampleSuperstore.csv")
    st.dataframe(df)
    agent = get_agent(df)

    if question := st.chat_input(placeholder="Escribe tu pregunta..."):
        st.session_state.agent_response = agent.invoke(
            {"input": question},
            config={"callbacks": [tracer]},
        )["output"]

    if st.session_state.agent_response:
        with st.chat_message("assistant"):
            st.write(st.session_state.agent_response)

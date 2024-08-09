import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from chat_model import ChatGemini

def main():
    st.set_page_config(page_title="LLM - Gemini Pro")
    st.title("AI-Powered ChatBot using LLM - Google Gemini Pro")

    model = ChatGemini()

    if "messages" not in st.session_state:
        # add greeting message to user
        st.session_state["messages"] = [
            AIMessage(content="Hello, how can I help you?")
        ]

    # if there are messages already in session, write them on app
    for message in st.session_state["messages"]:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)


    if prompt := st.chat_input("Say Something..."):
        if not isinstance(st.session_state["messages"][-1], HumanMessage):
            st.session_state["messages"].append(HumanMessage(content=prompt))
            message = st.chat_message("user")
            message.write(f"{prompt}")

        if not isinstance(st.session_state["messages"][-1], AIMessage):
            with st.chat_message("assistant"):
                # use .write() method for non-streaming, which means .invoke() method in chain
                response = st.write_stream(model.invoke_model(
                    query=prompt,
                     history=st.session_state["messages"]))
            st.session_state["messages"].append(AIMessage(content=response))



if __name__ == "__main__":
    main()
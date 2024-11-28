import streamlit as st
from chatbot_1.chatbot import qa,retriever

from openai import OpenAI
import streamlit as st

st.title("Medical AI Assistant")



if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar for additional controls
st.sidebar.title("Chatbot Settings")
st.sidebar.subheader("Conversation Management")
    
    # Clear chat history button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

    # Export chat history
if st.sidebar.button("📥 Export Conversation"):
    if st.session_state.messages:
        # Convert chat history to text
        chat_text = "\n\n".join([
            f"User: {msg['role']}\nAI: {msg['content']}" 
            for msg in st.session_state.messages
            ])
            
            # Create download button
        st.download_button(
            label="Download Conversation",
            data=chat_text,
            file_name="medical_chat_history.txt",
            mime="text/plain"
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
        # Get the last user message
            user_message = st.session_state.messages[-1]["content"]
            
            # Generate a response for the last user message
            response = retriever(qa, user_message)
            
            # Display the response
            st.markdown(response)

    # Append the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

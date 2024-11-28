import streamlit as st
from chatbot import retriever, qa


#st.write(retriever(qa,txt_area))

import streamlit as st
from streamlit_chat import message

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit app layout
st.title("Chatbot Interface")
st.markdown("Type your query below and press the arrow to submit!")

# Text area to display conversation history
chat_area = st.container()
with chat_area:
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            # Display user and chatbot messages alternately
            if i % 2 == 0:
                message(msg, is_user=True)  # User message
            else:
                message(msg)  # Bot response

# Input area at the bottom
input_area = st.container()
with input_area:
    col1, col2 = st.columns([9, 1])  # Adjust column widths
    with col1:
        user_input = st.text_input("Your Message", placeholder="Type your message here...", key="user_input")
    with col2:
        submit_button = st.button("➤", use_container_width=True)

# Handle user input
if submit_button and user_input.strip():
    # Append user input to session state
    st.session_state.messages.append(user_input)
    
    # Placeholder for chatbot response (replace with your model's response)
    bot_response = f"Echo: {user_input}"  # Replace with your model's logic
    st.session_state.messages.append(bot_response)
    
    # Clear the input box
    st.session_state.user_input = ""


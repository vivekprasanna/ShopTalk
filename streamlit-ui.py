import streamlit as st
import requests


API_ENDPOINT = "http://your-api-endpoint.com/chat"  # Replace with your actual API endpoint

st.title("🛍️ Shopping Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me about products:")

if st.button("Search"):
    if query:
        # Send request to the API
        response = requests.post(API_ENDPOINT,
                                 json={"question": query, "chat_history": st.session_state.chat_history})

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Sorry, I couldn't retrieve an answer.")
        else:
            answer = "Error: Unable to fetch response."

        # Update chat history
        st.session_state.chat_history.append((query, answer))
        st.write("**Assistant:**", answer)

# Display chat history
st.subheader("Chat History")
for q, a in st.session_state.chat_history:
    st.write(f"**You:** {q}")
    st.write(f"**Assistant:** {a}")
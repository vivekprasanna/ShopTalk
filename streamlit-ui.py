import streamlit as st
import requests
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
import os

API_ENDPOINT = "http://127.0.0.1:8080/api/submit"  # Replace with your actual API endpoint

st.title("🛍️ Shopping Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me about products:")
config = ConfigurationManager().get_data_validation_config()


def format_response(answer, retrieved_docs):
    """Format retrieved product details with images for display."""
    formatted_response = []

    for doc in retrieved_docs:
        description = doc.get("page_content", "No description available.")
        metadata = doc.get("metadata", {})
        name = metadata.get("name", "Unknown Product")
        image_path = metadata.get("image_path", "")
        image_url = f"{config.image_path_prefix}{image_path}" if image_path else ""
        # Append formatted response
        formatted_response.append({
            "name": name,
            "description": description,
            "image_path": image_url if os.path.exists(image_url) else ""
        })

    return formatted_response


if st.button("Search"):
    if query:
        # Send request to the API
        response = requests.post(API_ENDPOINT,
                                 json={"question": query, "chat_history": st.session_state.chat_history},
                                 headers={"Content-Type": "application/json"})
        logger.info(f"response to be displayed: {response.json()}")
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Sorry, I couldn't retrieve an answer.")
            retrieved_docs = data.get("retrieved_docs", [])  # Ensure API returns this field

            formatted_answer = format_response(answer, retrieved_docs)
        else:
            formatted_answer = [{"name": "Error", "description": "Unable to fetch response.", "image_url": ""}]

        # Update chat history
        st.session_state.chat_history.append((query, formatted_answer))

# Display chat history with images
st.subheader("Chat History")

for q, responses in st.session_state.chat_history:
    st.write(f"**You:** {q}")

    for response in responses:
        st.write(f"**🛍️ {response['name']}**)")
        st.write(f"✅ {response['description']}")
        if response["image_path"]:
            st.image(response["image_path"], caption=response["name"], use_container_width=True)
        else:
            st.write("🖼️ No image available")
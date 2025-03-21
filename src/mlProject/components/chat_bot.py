from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
# import subprocess

os.environ[
    "OPENAI_API_KEY"] = "your-key"

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

model_path = os.getenv("MODEL_DIR", "./fine_tuned_lora_tripletloss")  # Use environment variable
embeddings = HuggingFaceEmbeddings(model_name=model_path)

# Load embeddings and vector store
PERSIST_DIRECTORY = "./chromadb_vectorstore"
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)  # ✅ Updated import
retriever = vector_store.as_retriever()
config = ConfigurationManager().get_data_validation_config()

# result = subprocess.run(["ls", "-lh", "chromadb_vectorstore"], capture_output=True, text=True, check=True)
#
# logger.info(f"ls -lh chromadb_vectorstore: {result}")
#
#
# # Check stored collections
# print("Collections in Chroma:", vector_store.get())
#
# # Check if retriever has any documents
# docs = retriever.get_relevant_documents("red shoes")
# print("Number of retrieved documents:", len(docs))


# Use the updated OpenAI import
llm = OpenAI(temperature=0.7)

# Define a prompt for rephrasing user queries
rephrase_prompt = PromptTemplate.from_template(
    "Given the conversation history and the user’s latest question, rewrite the question for better retrieval.\n\n"
    "Chat History: {chat_history}\n"
    "User Question: {input}\n"
    "Rephrased Question:"
)

# Updated history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=rephrase_prompt  # ✅ Added missing required argument
)

# Define a retrieval-based answering prompt
qa_prompt = PromptTemplate.from_template(
    """Use the retrieved documents to answer the user's question. If you don't know, just say that you don't know.

    Retrieved documents: {context}

    Question: {input}

    Answer:
      - **🛍️ Product Name:** 
      - ✅ *Matching Details*
      - 🖼️ ![Thumbnail](Image_URL)
    """
)

# ✅ Create document combination chain
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# ✅ Updated retrieval chain with correct arguments
chatbot = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=combine_docs_chain  # ✅ Corrected argument
)


def ask_question(query, chat_history):
    """Function to handle user queries and maintain chat history."""
    retrieved_docs = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved docs: {retrieved_docs}")
    # qaresponse = qa_chain.run(input_documents=retrieved_docs, question=query)
    # print("QA Chain Response:", qaresponse)
    response = chatbot.invoke({"input": query, "chat_history": chat_history})
    formatted_response = format_response(response["answer"], retrieved_docs)
    chat_history.append((query, formatted_response))
    return response["answer"], chat_history, retrieved_docs


def format_response(answer, retrieved_docs):
    """Format the retrieved document details into a bulleted response with thumbnails."""
    formatted_answer = []
    for doc in retrieved_docs:
        name = doc.metadata.get("name", "Unknown Product")
        image_path = doc.metadata.get("image_path", "")
        image_url = f"{config.image_path_prefix}{image_path}" if image_path else ""

        formatted_answer.append(
            f"- **🛍️ {name}**\n"
            f"  - ✅ {doc.page_content.strip()}\n"
            f"  - 🖼️ ![Thumbnail]({image_url})\n"
        )
    return "\n".join(formatted_answer)
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from mlProject import logger
# import subprocess

os.environ[
    "OPENAI_API_KEY"] = "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"

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load embeddings and vector store
PERSIST_DIRECTORY = "./chromadb_vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="./fine_tuned_lora_tripletloss")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)  # ✅ Updated import
retriever = vector_store.as_retriever()

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

    Answer:"""
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
    logger.info(f"Collection count: {vector_store._collection.count()}")
    retrieved_docs = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved docs: {retrieved_docs}")
    # qaresponse = qa_chain.run(input_documents=retrieved_docs, question=query)
    # print("QA Chain Response:", qaresponse)
    response = chatbot.invoke({"input": query, "chat_history": chat_history})
    chat_history.append((query, response["answer"]))
    return response["answer"], chat_history

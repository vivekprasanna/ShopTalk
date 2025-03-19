from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd


def batch(iterable, size):
    """Helper function to split data into batches."""
    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]

def main():
    # Define the storage path
    PERSIST_DIRECTORY = "chromadb_vectorstore"

    # Initialize ChromaDB and OpenAI embeddings
    embeddings = HuggingFaceEmbeddings(model_name="./app/fine_tuned_lora_tripletloss")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)  # Pass embeddings to Chroma

    df_sample = pd.read_csv("artifacts/data_validation/df_sample_20k.csv")

    texts = []
    metadatas = []
    embeddings = []

    for _, row in df_sample.iterrows():
        text = row["complete_product_description"]  # Product description

        texts.append(text)
        metadatas.append({"id": str(row["item_id"]),
                          "name": row["item_name"],
                          "category": row["product_type"],
                          "country": row["country"],
                          "image_path": row["image_path"]})

    BATCH_SIZE = 5000  # Keep it below 5461 to avoid errors

    # Split your data into chunks before inserting
    for text_batch, meta_batch in zip(batch(texts, BATCH_SIZE), batch(metadatas, BATCH_SIZE)):
        vectorstore.add_texts(texts=text_batch, metadatas=meta_batch)
        vectorstore.persist()  # Ensure it's saved after each batch

    # vectorstore.add_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    # vectorstore.persist()

    print("Embeddings generated and vector store saved to:", PERSIST_DIRECTORY)
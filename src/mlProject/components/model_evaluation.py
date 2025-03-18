import pandas as pd
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import get_device, save_json
from transformers import AutoModel, AutoTokenizer
import torch
from peft import PeftModel
import chromadb
from mlProject import logger
import numpy as np
from pathlib import Path
import mlflow
from urllib.parse import urlparse
import dagshub

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def get_embedding(self, model, tokenizer, text, device):
        model.to(device)
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure tensors are on MPS)
        with torch.no_grad():
            output = model(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation

    def compute_precision_at_k(self, query_embedding, category, collection, test_data, k=10):
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        retrieved_ids = results["ids"][0]

        # Count how many retrieved items belong to the same category
        filtered_data = test_data[test_data["item_id"].isin(retrieved_ids)]
        relevant_count = (filtered_data["product_type"] == category).sum()

        return relevant_count / k  # Precision@K formula

    def compute_recall_at_k(self, query_embedding, category, collection, test_data, k=10):
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        retrieved_ids = results["ids"][0]

        # Get all relevant product IDs from the same category
        relevant_items = test_data[test_data["product_type"] == category]["item_id"].tolist()

        # Count how many retrieved items are relevant
        relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_items)

        total_relevant = len(relevant_items)

        return relevant_retrieved / min(total_relevant, k)  # Normalize by available relevant items

    # Function to compute mAP
    def compute_map(self, query_embedding, category, collection, test_data, k=10):
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        retrieved_ids = results["ids"][0]

        relevant_items = test_data[test_data["product_type"] == category]["item_id"].tolist()
        num_relevant = len(relevant_items)

        precisions = []
        relevant_count = 0

        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_items:
                relevant_count += 1
                precisions.append(relevant_count / i)

        return sum(precisions) / min(num_relevant, k) if precisions else 0.0

    # Function to compute NDCG@K
    def compute_ndcg_at_k(self, query_embedding, category, collection, test_data, k=10):
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        retrieved_ids = results["ids"][0]

        relevant_items = test_data[test_data["product_type"] == category]["item_id"].tolist()

        # Compute DCG@K
        dcg = sum((1 / np.log2(i + 1)) if rid in relevant_items else 0 for i, rid in enumerate(retrieved_ids, start=1))

        # Compute Ideal DCG@K (perfect ranking)
        idcg = sum((1 / np.log2(i + 1)) for i in range(1, min(len(relevant_items), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self):
        dagshub.init(repo_owner='vivekprasanna.prabhu', repo_name='ShopTalk', mlflow=True)
        saved_model_path = "fine_tuned_lora_tripletloss"
        base_model = AutoModel.from_pretrained(saved_model_path)
        peft_model = PeftModel.from_pretrained(base_model, saved_model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        device = get_device()
        if device.type == "mps":
            peft_model = peft_model.to(torch.float32)
        else:
            peft_model = peft_model.to(device)

        with mlflow.start_run():

            tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

            # Ensure tokenizer is prepared correctly
            # Set the padding token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token (if available)
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens(
                        {"pad_token": "[PAD]"})  # Add a new pad token if eos_token is also missing

            test_data = pd.read_csv(self.config.test_data_path)
            test_data["embedding"] = test_data["complete_product_description"].apply(
                lambda x: self.get_embedding(peft_model, tokenizer, x, device))

            # Initialize ChromaDB client and create a collection
            chroma_client = chromadb.PersistentClient(path="artifacts/model_evaluation/chroma_db")  # Persistent storage
            collection = chroma_client.get_or_create_collection(name="product_embeddings")

            # Insert embeddings into ChromaDB, item_id,item_name,product_type,country,enhanced_product_desc,image_path
            for idx, row in test_data.iterrows():
                collection.add(
                    ids=[row["item_id"]],
                    embeddings=row["embedding"],
                    metadatas=[{"index": idx, "item_id": row["item_id"], "item_name": row["item_name"], "country": row["country"], "image_path": row["image_path"]}]
                )
            logger.info("Embeddings stored in ChromaDB!")

            # Compute Precision@K for all products
            test_data["precision_at_k"] = test_data.apply(
                lambda row: self.compute_precision_at_k(row["embedding"], row["product_type"], collection, test_data, k=10),
                axis=1
            )
            test_data["recall_at_k"] = test_data.apply(
                lambda row: self.compute_recall_at_k(row["embedding"], row["product_type"], collection, test_data, k=10),
                axis=1
            )
            test_data["mAP"] = test_data.apply(
                lambda row: self.compute_map(row["embedding"], row["product_type"], collection, test_data, k=10),
                axis=1
            )
            test_data["NDCG"] = test_data.apply(
                lambda row: self.compute_ndcg_at_k(row["embedding"], row["product_type"], collection, test_data, k=10),
                axis=1
            )

            # Display results
            logger.info(test_data[["item_id", "product_type", "precision_at_k", "recall_at_k", "mAP", "NDCG"]])
            selected_columns = ["precision_at_k", "recall_at_k", "mAP", "NDCG"]
            eval_result = test_data[selected_columns]
            avg_precision = eval_result["precision_at_k"].mean()
            avg_recall = eval_result["recall_at_k"].mean()
            avg_map = eval_result["mAP"].mean()
            avg_ndcg = eval_result["NDCG"].mean()
            scores = {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_mAP": avg_map, "avg_NDCG": avg_ndcg}
            logger.info(f"Avg scores: {scores}")
            save_json(Path(self.config.metric_file_name), data=scores)
            test_data.to_csv("artifacts/model_evaluation/eval_result.csv")

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("Mean_precision_at_k_10", avg_precision)
            mlflow.log_metric("Mean_recall_at_k_10", avg_recall)
            mlflow.log_metric("Mean_mAP", avg_map)
            mlflow.log_metric("Mean_NDCG", avg_ndcg)

            mlflow.sklearn.log_model(peft_model, "model", registered_model_name="fine_tuned_lora_triplet_loss")




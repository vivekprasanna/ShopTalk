from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
from peft import PeftModel
from mlProject.utils.common import get_device
from mlProject import logger


class LoRAEmbeddings(Embeddings):

    def __init__(self):
        self.saved_model_path = "fine_tuned_lora_tripletloss"
        self.base_model = AutoModel.from_pretrained(self.saved_model_path)
        self.peft_model = PeftModel.from_pretrained(self.base_model, self.saved_model_path)
        self.device = get_device()

        # Ensure model weights are in float32 on MPS
        if self.device.type == "mps":
            self.peft_model = self.peft_model.to(torch.float32)
        else:
            self.peft_model = self.peft_model.to(self.device)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.saved_model_path)

        # Ensure tokenizer is prepared correctly
        # Set the padding token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token (if available)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add a new pad token if eos_token is also missing

        logger.info("PEFT Model and Tokenizer Loaded Successfully!")

    def custom_embedding_function(self, text):
        # print(f"Embedding text: '{text[:50]}...'")  # Debug: Check if function is called
        """Compute embedding using fine-tuned LoRA model."""
        tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)

        # Move tokenized input to the same device as the model
        tokenized = {key: value.to(self.device, dtype=torch.long) for key, value in tokenized.items()}

        with torch.no_grad():
            # Forward pass through the model (on the correct device)
            outputs = self.peft_model(**tokenized)
            # print(f"Type of outputs: {type(outputs)}") #debug
            # print(f"Outputs keys: {getattr(outputs, 'keys', lambda: None)()}") #debug
            # print(f"Outputs shape: {getattr(outputs, 'shape', lambda: None)}") #debug
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output.cpu().numpy().tolist()[0]
            elif hasattr(outputs, "last_hidden_state"):
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]
                # OR CLS token extraction
                # embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]
            else:
                raise ValueError(f"Unexpected output structure: {type(outputs)}")

        # print(f"Embedding length: {len(embeddings)}, Type: {type(embeddings)}") #debug
        return embeddings

    def embed_documents(self, texts):
        logger.info(f"Number of texts to embed: {len(texts)}")  # Debug: Check number of texts
        embeddings = []
        for text in texts:
            embeddings.append(self.custom_embedding_function(text))
        logger.info(f"Number of embeddings created: {len(embeddings)}")  # Debug: Check number of embeddings
        return embeddings

    def embed_query(self, text):
        return self.custom_embedding_function(text)  # use same embedding function for query.
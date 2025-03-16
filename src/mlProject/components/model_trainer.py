import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig, TaskType
import random
from mlProject.components.ProductTripletDataset import ProductTripletDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import TripletMarginLoss
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger


def get_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device: ", device)
    return device


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        model_name = self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        df_tune_sample = pd.read_csv(self.config.data_path)
        category_groups = df_tune_sample.groupby('product_type')

        logger.info("Initializing data for training")

        # Create triplet samples
        triplets = []
        for category, group in category_groups:
            group_list = group.to_dict("records")

            for anchor in group_list:
                # Select a positive sample from the same category
                positive = random.choice(group_list)

                # Select a negative sample from a different category
                while True:
                    negative_category = random.choice(df_tune_sample["product_type"].unique())
                    if negative_category != category:
                        negative = \
                        df_tune_sample[df_tune_sample["product_type"] == negative_category].sample(1).to_dict(
                            "records")[0]
                        break

                triplets.append(
                    (anchor["item_name"], positive["enhanced_product_desc"], negative["enhanced_product_desc"]))

        logger.info("Created triplet data for training")

        # Apply LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )

        logger.info(f"Created LoRA config: {lora_config}")

        # Wrap the model with LoRA
        peft_model = get_peft_model(base_model, lora_config)

        train_dataset = ProductTripletDataset(triplets, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        device = get_device()  # Fix function call
        peft_model.to(device)

        # Ensure model weights are in float32 on MPS
        if device == "mps":
            peft_model = peft_model.to(torch.float32)

        optimizer = optim.AdamW(peft_model.parameters(), lr=5e-5)
        loss_fn = TripletMarginLoss(margin=0.5)

        # Training Loop
        for epoch in range(5):
            peft_model.train()
            total_loss = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                # Convert input tensors to the correct device
                dtype = torch.float16 if device == "cuda" else torch.float32

                anchor_inputs = {key: value.squeeze().to(device, dtype=torch.long) for key, value in
                                 batch["anchor"].items()}
                positive_inputs = {key: value.squeeze().to(device, dtype=torch.long) for key, value in
                                   batch["positive"].items()}
                negative_inputs = {key: value.squeeze().to(device, dtype=torch.long) for key, value in
                                   batch["negative"].items()}

                # Get embeddings
                anchor_embed = peft_model(**anchor_inputs).last_hidden_state[:, 0, :]
                positive_embed = peft_model(**positive_inputs).last_hidden_state[:, 0, :]
                negative_embed = peft_model(**negative_inputs).last_hidden_state[:, 0, :]

                # Compute loss
                loss = loss_fn(anchor_embed, positive_embed, negative_embed)

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

        # Save fine-tuned model
        peft_model.save_pretrained("fine_tuned_lora_tripletloss")
        peft_model.base_model.save_pretrained("fine_tuned_lora_tripletloss")
        tokenizer.save_pretrained("fine_tuned_lora_tripletloss")
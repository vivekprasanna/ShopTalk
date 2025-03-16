from torch.utils.data import Dataset


class ProductTripletDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        anchor_input = self.tokenizer(anchor, padding="max_length", truncation=True, return_tensors="pt",
                                      max_length=128)
        positive_input = self.tokenizer(positive, padding="max_length", truncation=True, return_tensors="pt",
                                        max_length=128)
        negative_input = self.tokenizer(negative, padding="max_length", truncation=True, return_tensors="pt",
                                        max_length=128)

        return {
            "anchor": anchor_input,
            "positive": positive_input,
            "negative": negative_input
        }

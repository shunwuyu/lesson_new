import torch
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('./data/ChnSentiCorp')[split]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        return text, label
    
dataset = Dataset('train')
print(len(dataset), dataset[0])
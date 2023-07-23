from transformers import BertTokenizer
from transformers import BertModel
from datasets import load_from_disk
from transformers import AdamW
token = BertTokenizer.from_pretrained('bert-base-chinese')
import torch

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

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True, # 截断
        padding='max_length',
        max_length=500,
        return_tensors='pt', # 返回pytorch tensor
        return_length=True, 
    )
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels) 
    return input_ids, attention_mask, token_type_ids, labels

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print(len(loader)) 
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
print(token)

pretrained = BertModel.from_pretrained('bert-base-chinese')

for param in pretrained.parameters():
    param.requires_grad_(False)
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)

print(out.last_hidden_state.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
    
    def forward(self,  input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            out = self.fc(out.last_hidden_state[:, 0])
            return out

model = Model()
print(model(input_ids = input_ids,
       attention_mask = attention_mask,
         token_type_ids = token_type_ids).shape)

import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW



pretained = BertModel.from_pretrained('bert-base-chinese')

for param in pretained.parameters(): 
    param.requires_grad_(False)

token = BertTokenizer.from_pretrained('bert-base-chinese')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        dataset = load_from_disk('./data/ChnSentiCorp')[split]
        def f(data):
            return len(data['text']) > 30
        self.dataset = dataset.filter(f)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        return text
    
dataset = Dataset('train')
# print(len(dataset), dataset[0])

def collate_fn(data):
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=data,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt',
        return_length=True,
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids'] 

    labels = input_ids[:, 15].reshape(-1).clone() 
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

    return input_ids, attention_mask, token_type_ids, labels

loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=16, 
    collate_fn=collate_fn, 
    shuffle=True, 
    drop_last=True,
)

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader): 
    break

# print(len(loader))
# print(token.decode(input_ids[0]))
# print(token.decode(labels[0]))
# print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels.shape)

out = pretained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
# print(out.last_hidden_state.shape)
# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, token.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 
        with torch.no_grad():
            out = pretained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out = self.fc(out.last_hidden_state[:, 15])
            return out
        
model = Model()
model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).shape

optimizer = AdamW(params=model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss() 
model.train()

for epoch in range(5):
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            out = out.argmax(dim=-1) 
            accuracy = (out == labels).sum().item() / len(labels)
            print(epoch, i, loss.item(), accuracy)
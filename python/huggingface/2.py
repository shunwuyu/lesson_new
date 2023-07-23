# 加载数据集
from datasets import load_from_disk
# 情感分析数据集
dataset = load_from_disk('./data/ChnSentiCorp')
# print(dataset['train'][:10])
dataset = dataset['train']
# sorted_dataset = dataset.sort('label')
# # print(sorted_dataset[-10:])
# shuffled_dataset = sorted_dataset.shuffle(seed=42)
# print(shuffled_dataset['label'][:10])

def f(data):
    return data['text'].startswith('选择')

start_with_ar = dataset.filter(f)

print(len(start_with_ar), start_with_ar['text'])
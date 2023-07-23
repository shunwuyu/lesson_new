from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因是方便。',
    '笔记本的键盘确定爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷，真有点郁闷。',
    '机器背面似乎被 撕了张什么标签，残胶还在。'
]

# 一次编码两个句子
# out = tokenizer.encode(
#     text = sents[0],
#     text_pair=sents[1],
#     truncation=True, 
#     padding='max_length',
#     add_special_tokens=True, 
#     max_length=30,
#     return_tensors=None,
# )

# 增强的编码方式
# out = tokenizer.encode_plus(
#     text=sents[0],
#     text_pair=sents[1],
#     truncation=True,
#     padding='max_length',
#     max_length=30,
#     add_special_tokens=True,
#     # tf pt np list 
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True,
# )


# print(out)
# print(tokenizer.decode(out))



# tokenizer.decode(out['input_ids'])
# 批量编码句子
# out = tokenizer.batch_encode_plus(
#     batch_text_or_text_pairs=[sents[0], sents[1]],
#     add_special_tokens=True,
#     truncation=True,
#     padding='max_length',
#     max_length=15,
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True,
# )

# out = tokenizer.batch_encode_plus(
#     batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
#     add_special_tokens=True,
#     truncation=True,
#     padding='max_length',
#     max_length=15,
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True,
# )

# for k, v in out.items():
#     print(k, ':', v)

# zidian = tokenizer.get_vocab() 
# print(type(zidian), len(zidian), '月光' in zidian)

tokenizer.add_tokens(new_tokens=['月光', '希望'])
zidian = tokenizer.get_vocab()
print(type(zidian), len(zidian),zidian['月'],zidian['月光'], '月光' in zidian)

out = tokenizer.encode_plus(
    text='月光的新希望[EOS]',
    text_pair=None,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    return_tensors=None,
)

print(out)
print(tokenizer.decode(out['input_ids']))
from transformers import pipeline
# 文本分类
classifier = pipeline("sentiment-analysis")
result = classifier("I hate you")[0]
print(result)
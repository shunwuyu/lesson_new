from transformers import pipeline
unmasker = pipeline('fill-mask')
from pprint import pprint 

sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks.'
pprint(unmasker(sentence))
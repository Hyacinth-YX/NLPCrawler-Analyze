import pytorch_pretrained_bert as bert

import torch

bert_model_dir = "bert_model/bert-chinese/"

tokenizer = bert.BertTokenizer.from_pretrained(bert_model_dir)

bert = bert.BertModel.from_pretrained(bert_model_dir )

with open("Corpus/3D技术的前世今生.txt",encoding="utf-8") as f:

	s = f.read()

print(s)

tokens = tokenizer.tokenize(s)

print(tokens)

ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

print(ids)

result = bert(ids, output_all_encoded_layers=False)


print(result)
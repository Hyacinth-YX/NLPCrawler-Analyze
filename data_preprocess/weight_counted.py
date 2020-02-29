import config as args
from collections import Counter
import json
import numpy
def get_weight_map(dir):
	labels = []
	with open(dir,encoding = "utf-8") as f:
			lines = f.readlines()
			for line in lines:
				line = json.loads(line)
				labels += line["target"].split(" ")
	count_dict = dict(Counter(labels))
	total = sum(count_dict.values())
	keys = list(count_dict.keys())
	keys.sort(key=lambda key:count_dict[key],reverse=True)
	sorted_count = dict([key,count_dict[key]]for key in keys)
	for key in sorted_count:
		sorted_count[key] = numpy.log(total/sorted_count[key])
	return sorted_count
weight_map = get_weight_map(args.SRL_train)
import torch
print(weight_map)
labels = ['O','I-ARGM-ADV', 'I-ARGM-TMP', 'B-ARG0', 'B-ARG1', 'B-ARGM-TMP', 'I-ARG0', 'B-ARGM-ADV', 'I-ARG1']
weight = torch.FloatTensor([weight_map[label] for idx,label in enumerate(labels)])
print(weight)


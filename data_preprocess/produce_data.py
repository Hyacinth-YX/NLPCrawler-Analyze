from utils.loggerUtils import init_logger
from utils.corpusUtils import CTB
from utils.dataprocessUtils import train_val_split,sent2char
import config.args as args
from  tqdm import tqdm
logger = init_logger("Produce data",args.log_path)
import json
import os

def produce_NER_data():

	for set_idx,dir in enumerate([args.NER_TRAIN_SOURCE,args.NER_VALID_SOURCE]):

		if set_idx == 0:
			subset = "train"
		else:
			subset = "dev"

		logger.info("Produce NER data : %s"%subset)

		with open(dir, "r", encoding='utf-8') as fr:
			lines = []
			for line in fr:
				_line = line.strip('\n')
				lines.append(_line)

		label_list = set()

		for idx in tqdm(range(len(lines)),desc = "Produce File"):
			line = json.loads(lines[idx])
			text_a = line["text"]
			labels = ["O"] * len(text_a)
			label_spans = line["label"]
			for label in label_spans:
				textvalue_spans = label_spans[label]
				spans = list(textvalue_spans.values())[0]
				for span in spans:
					for j in range(span[0], span[1] + 1):
						if j == span[0]:
							labels[j] = "B-"+label
						else:
							labels[j] = "I-"+label
						label_list.add(labels[j])
			source = " ".join(text_a)
			target = " ".join(labels)
			df = {"source":source,"target":target}

			with open(os.path.join(args.NER_data_dir,subset + ".json"),"a",encoding = "utf-8") as f:
				f.write(json.dumps(df) + "\n")

		logger.info("-------Produced data Example--------")
		logger.info("source: %s" % source)
		logger.info("target: %s" % target)
		logger.info("label list " + str(label_list))

def produce_predict_seqlabeling_data():

	logger.info("Produce SRL_predict data" )
	CTB_corpus =  CTB(load_cpb=True)

	sources = []
	targets = []
	labels = set()
	for file_name in CTB_corpus.files:
		file = CTB_corpus.files[file_name]
		sents = file.sents
		for idx,sent in enumerate(sents):

			PA_structures = sent["PA-structure"]
			terminals = sent["parseTree"].terminals
			word_labels = ["O" for i in range(len(terminals))]
			for PA in PA_structures:
				predict_idx   = PA["predict"]
				predict_label = PA["predict_type"]
				word_labels[predict_idx] = predict_label

			source =""
			target = []
			for index,terminal in enumerate(terminals):
				word_label = word_labels[index]
				consitute_word = terminal.split(" ")
				if len(consitute_word) == 2:
					consitute,word = consitute_word
					if consitute != "-NONE-":
						source += word
						word_len = len(word)
						if word_label == "O":
							target += ["O" for p in range(word_len)]
							labels.add("O")
						else:
							for p in range(word_len):
								if p == 0 :
									target.append("B-" + word_label)
								else:
									target.append("I-" + word_label)
								labels.add("B-" + word_label)
								labels.add("I-" + word_label)
			source = " ".join(source)
			target  = " ".join(target)
			assert len(source.split()) == len(target.split())
			sources.append(source)
			targets.append(target)

	logger.info("split data into train & dev")

	train,valid = train_val_split(sources,targets)

	with open(args.SRL_predict_train,"a",encoding = "utf-8") as f:
		with open(args.SRL_predict_valid,"a",encoding = "utf-8") as f2:
			for i in tqdm(range(len(train)),desc = "write to train set"):
					df = {"source":train[i][0],"target":train[i][1]}
					f.write(json.dumps(df) + "\n")
			for j in tqdm(range(len(valid)),desc = "write to valid set"):
					df = {"source":valid[j][0],"target":valid[j][1]}
					f2.write(json.dumps(df) + "\n")

	logger.info("Produce SRL_predict data done !")
	logger.info("Train: %d examples , Dev: %d examples " % (len(train),len(valid)))
	logger.info("-------Produced data Example--------")
	logger.info("source: %s" % train[0][0])
	logger.info("target: %s" % train[0][1])
	logger.info("label list :"  + str(labels) )
def produce_PA_structure_data():

	logger.info("Produce SRL_PA_structure data" )
	CTB_corpus =  CTB(load_cpb=True)

	X = []
	seqLabel = []
	labels = set()
	for file_name in CTB_corpus.files:
		file = CTB_corpus.files[file_name]
		sents = file.sents[3:]
		for idx,sent in enumerate(sents):
			PA_structures = sent["PA-structure"]
			terminals = sent["parseTree"].terminals
			args_word_labels = [["O" for j in range(len(terminals))] for i in range(len(PA_structures))] #每个PA对应的论元标签序列
			args_predicts = []
			#先对每个谓词创建该句的论元标签序列
			for idx,PA in enumerate(PA_structures):
				#便利每句话的每个PA
				arguments = PA["args"]
				for argument in arguments:
					spans = arguments[argument]
					spans = spans.replace("*",",")
					spans = spans.split(",")
					for span in spans:
						span = span.split(":")
						for p in range(int(span[0]), int(span[0]) + int(span[1]) + 1):
							try:
								args_word_labels[idx][p] = argument
							except:
								continue
				args_predicts.append(terminals[PA["predict"]].split(" ")[1])

			source = []
			targets = [[] for i in range(len(PA_structures))]

			for index,terminal in enumerate(terminals):
				consitute_word = terminal.split(" ")
				if len(consitute_word) != 2:
					continue
				consitute,word = consitute_word
				if consitute == "-NONE-":
					continue
				word_len = len(word)
				for p in range(word_len):
					source.append(word[p])
				#print(source)
				for idx,word_labels in enumerate(args_word_labels):
					target = targets[idx]#第idx 个论元结构的label序列存储对象
					word_label = word_labels[index] # 这个单词对应的标签
					#print("args",idx,"coresponding label",word_label)
					for j in range(word_len):
						if word_label != "O":#需要IB标签
							#判断是否为B
							if j == 0 and (target == [] or word_label not in target[-1]):
								encoded_word_label = "B-" + word_label
							else:
								encoded_word_label = "I-" + word_label
						else:
							encoded_word_label = word_label
						labels.add(encoded_word_label)
						target.append(encoded_word_label)

			source = " ".join(source)
			for i in range(len(targets)):
				targets[i] = " ".join(targets[i])
				args_predicts[i] = " ".join(args_predicts[i])
				assert len(targets[i].split()) == len(source.split())
				X.append((source,args_predicts[i]))
				seqLabel.append(targets[i])


	logger.info("split data into train & dev")

	train,valid = train_val_split(X,seqLabel)

	with open(args.SRL_train,"a",encoding = "utf-8") as f:
		with open(args.SRL_valid,"a",encoding = "utf-8") as f2:
			for i in tqdm(range(len(train)),desc = "write to train set"):
					df = {"source":train[i][0],"target":train[i][1]}
					f.write(json.dumps(df) + "\n")
			for j in tqdm(range(len(valid)),desc = "write to valid set"):
					df = {"source":valid[j][0],"target":valid[j][1]}
					f2.write(json.dumps(df) + "\n")

	logger.info("Produce SRL_predict data done !")
	logger.info("Train: %d examples , Dev: %d examples " % (len(train),len(valid)))
	logger.info("-------Produced data Example--------")
	logger.info("source: %s" % str(train[0][0]))
	logger.info("target: %s" % str(train[0][1]))
	#logger.info("label list :"  + str(labels) )

def produce_NER_PEOPLE_DAILY_data(stop_word_list=None, vocab_size=None):
	"""实际情况下，train和valid通常是需要自己划分的，这里将train和valid数据集划分好写入文件"""
	targets, sentences = [],[]
	with open("data3\source_BIO_2014_cropus.txt", 'r',encoding = "utf-8") as fr_1, \
			open("data3\\target_BIO_2014_cropus.txt", 'r',encoding = "utf-8") as fr_2:
		for sent, target in tqdm(zip(fr_1, fr_2), desc='text_to_id'):
			chars = sent2char(sent)
			label = sent2char(target)
			targets.append(label)
			sentences.append(chars)
	train, valid = train_val_split(sentences, targets)
	with open(args.NER_PEOPLEDAILY_TRAIN, 'w') as fw:
		for sent, label in train:
			sent = ' '.join([str(w) for w in sent])
			label = ' '.join([str(l) for l in label])
			df = {"source": sent, "target": label}
			encode_json = json.dumps(df)
			print(encode_json, file=fw)
		logger.info('Train set write done')
	with open(args.NER_PEOPLEDAILY_VALID, 'w') as fw:
		for sent, label in valid:
			sent = ' '.join([str(w) for w in sent])
			label = ' '.join([str(l) for l in label])
			df = {"source": sent, "target": label}
			encode_json = json.dumps(df)
			print(encode_json, file=fw)
		logger.info('Dev set write done')


if __name__ == "__main__":

	produce_PA_structure_data()
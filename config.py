from typing import Optional,List,Union

class ProjectConfiguration:

	#configure path of this project
	ROOT_DIR: str    = r"D:\git_of_skydownacai\\NLP_PROJECT\\"
	data_path: str   = ROOT_DIR + r"data/"
	tempfile: str    = ROOT_DIR + r"temp/"
	model_path: str	= ROOT_DIR + r"models/"
	log_path: str  	= ROOT_DIR + "logs/"
	embedding_path: str  = ROOT_DIR + "embeddings/"
	trained_model_path: str = ROOT_DIR + "fine_tuned_model/"

	#Bert Model Related
	bert_embedding_path: str = embedding_path + "bert_embedding.txt"
	bert_model_dir: str =ROOT_DIR +  r"bert_model/bert-chinese"
	bert_vocab: str = bert_model_dir + "/vocab.txt"

	#Trained model
	trained_NER_CLUE: str = trained_model_path + "BERT_NER_CLUE.bin"
	trained_NER_PEOPLEDAILY: str = trained_model_path + "BERT_NER_PEOPLEDAILY.bin"
	trained_PredictLabeling: str = trained_model_path + "BERT_PREDICT_LABELING.bin"
	trained_SRL: str = trained_model_path + "BERT_SRL.bin"
	trained_Pointer_Network: str = trained_model_path + "seq2seq.04"

	#ROUGE
	rouge_script_path = ROOT_DIR + "tools/rouge/"

	#OUTPUT
	output_path = ROOT_DIR + "output/"
class Corpus_Path:

	CTB7_dir: str = "D:/ctb/7/data/"
	CTB9_dir: str = "D:/ctb/9/data/"
	CPB_dir: str = "D:/cpb/data/"
	CPB_verb_file: str = CPB_dir + "cpb3.0-verbs.txt"
	CPB_nouns_file: str = CPB_dir + "cpb3.0-nouns.txt"
	textsummary_file = 'D:/nlpCorpus/nlpcc2017textsummarization/train_with_summ.txt'

class NER_Param:
	model_dir: str = ProjectConfiguration.model_path + r"BERT_NER/"
	data_dir: str  = ProjectConfiguration.data_path + r"NER_CLUE/"

	#NER data (dataset: CLUE2020) file path
	source_data_dir: str = r"D:/nlpCorpus/CLUE/dataset/NER/"
	train_source: str = source_data_dir + r"train.json"
	valid_source: str = source_data_dir + r"dev.json"
	test_source: str  = source_data_dir + r"test.json"
	train: str = data_dir + r"train.json"
	valid: str = data_dir + r"dev.json"
	test: str  = data_dir + r"test.json"
	labels: List[str] = ['O', 'B-address', 'I-address', 'B-book', 'I-book', 'B-company', 'I-company', 'B-game', 'I-game', 'B-government', 'I-government', 'B-movie', 'I-movie', 'B-name', 'I-name', 'B-organization', 'I-organization', 'B-position', 'I-position', 'B-scene', 'I-scene']

	#NER data (dataset: People Daily) file path
	PEOPLEDAILY_source_dir: str = "data3/source_BIO_2014_cropus.txt"
	PEOPLEDAILY_target_dir: str = "data3/target_BIO_2014_cropus.txt"
	PEOPLEDAILY_dir: str = ProjectConfiguration.data_path + r"NER_PeopleDaily/"
	PEOPLEDAILY_train: str =  PEOPLEDAILY_dir+ r"train.json"
	PEOPLEDAILY_valid: str =  PEOPLEDAILY_dir+ r"dev.json"
	PEOPLEDAILY_labels: List[str] = [ "O","B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC",]

class PredictLabeling_Param:

	data_dir: str = ProjectConfiguration.data_path + r"SRL_Predict/"
	label: List[str] = ['O','I-norminalVerb', 'B-norminalVerb', 'I-verb', 'B-verb']
	train: str = data_dir + "train.json"
	valid: str = data_dir + "dev.json"
	model_dir: str = ProjectConfiguration.model_path + r"BERT_PredictLabeling/"

class SRL_Param:

	data_dir: str = ProjectConfiguration.data_path + r"SRL_PA/"
	train: str = data_dir + "train.json"
	valid: str = data_dir + "dev.json"
	model_dir: str = ProjectConfiguration.model_path + r"BERT_SRL/"
	label: List[str] = ['O','I-ARG0-PSE', 'I-ARGM-BNF', 'I-ARGM-ADV', 'B-ARGM-NEG', 'B-ARGM-PRP', 'B-ARGM-EXT', 'B-ARG0-PSE', 'I-ARGM-DGR', 'B-ARGM-FRQ', 'I-ARGM-FRQ', 'I-ARGM-LOC', 'B-ARGM-BNF', 'I-ARGM-TPC', 'B-ARG0', 'I-ARGM-DIR', 'I-ARGM-TMP', 'B-ARGM-MNR', 'I-ARG3', 'B-ARG2', 'B-ARGM-TMP', 'I-ARGM-PRP', 'I-ARGM-DIS', 'I-ARGM-MNR', 'B-ARGM-DGR', 'B-ARGM-DIR', 'B-ARG3', 'B-ARGM-TPC', 'I-ARG1', 'B-ARGM-DIS', 'B-ARG1', 'I-ARG0', 'B-ARGM-ADV', 'I-ARG2', 'B-ARGM-CND', 'I-ARGM-CND', 'I-ARGM-NEG', 'I-ARGM-EXT', 'B-ARGM-LOC']

class BertTokenClassfierParam:
	# Training
	device: Union[str,List[str]]= "cuda:0"
	gradient_accumulation_steps: str = 1
	train_batch_size: int = 5
	eval_batch_size: int = 5
	learning_rate: int = 2e-5
	num_train_epochs: int = 6
	warmup_proportion: int = 0.1
	no_cuda: bool = False
	loss_scale: int = 0
	fp16: bool = False

class Seq2SeqPram:

	device: Union[str,List[str]]= "cpu"
	data_file: str = ProjectConfiguration.data_path  + r"text_summary/"
	model_dir: str = ProjectConfiguration.model_path + r"text_summary/"
	train: str = data_file + "train.txt"
	valid: str = data_file + "valid.txt"
	vocab_size: int = 50000
	vocab: str = data_file + "vocab.json"
	hidden_size: int = 150  # of the encoder; default decoder size is doubled if encoder is bidi
	dec_hidden_size: Optional[int] = 200  # if set, a matrix will transform enc state into dec state
	embed_size: int = 100
	enc_bidi: bool = True
	enc_attn: bool = True  # decoder has attention over encoder states?
	dec_attn: bool = False  # decoder has attention over previous decoder states?
	pointer: bool = True  # use pointer network (copy mechanism) in addition to word generator?
	out_embed_size: Optional[int] = None  # if set, use an additional layer before decoder output
	tie_embed: bool = True  # tie the decoder output layer to the input embedding layer?

	# Coverage (to turn on/off, change both `enc_attn_cover` and `cover_loss`)
	enc_attn_cover: bool = True  # provide coverage as input when computing enc attn?
	cover_func: str = 'max'  # how to aggregate previous attention distributions? sum or max
	cover_loss: float = 1  # add coverage loss if > 0; weight of coverage loss as compared to NLLLoss
	show_cover_loss: bool = False  # include coverage loss in the loss shown in the progress bar?

	# Regularization
	enc_rnn_dropout: float = 0
	dec_in_dropout: float = 0
	dec_rnn_dropout: float = 0
	dec_out_dropout: float = 0

	# Training
	optimizer: str = 'adam'  # adam or adagrad
	lr: float = 0.001  # learning rate
	adagrad_accumulator: float = 0.1
	lr_decay_step: int = 5  # decay lr every how many epochs?
	lr_decay: Optional[float] = None  # decay lr by multiplying this factor
	batch_size: int = 30
	n_batches: int = 400  # how many batches per epoch
	val_batch_size: int = 20
	n_val_batches: int = 200  # how many validation batches per epoch
	n_epochs: int = 75
	pack_seq: bool = True  # use packed sequence to skip PAD inputs?
	forcing_ratio: float = 0.75  # initial percentage of using teacher forcing
	partial_forcing: bool = True  # in a seq, can some steps be teacher forced and some not?
	forcing_decay_type: Optional[str] = 'exp'  # linear, exp, sigmoid, or None
	forcing_decay: float = 0.9999
	sample: bool = True  # are non-teacher forced inputs based on sampling or greedy selection?
	grad_norm: float = 1  # use gradient clipping if > 0; max gradient norm
	# note: enabling reinforcement learning can significantly slow down training
	rl_ratio: float = 0.2  # use mixed objective if > 0; ratio of RL in the loss function
	rl_ratio_power: float = 1  # increase rl_ratio by **= rl_ratio_power after each epoch; (0, 1]
	rl_start_epoch: int = 5  # start RL at which epoch (later start can ensure a strong baseline)?

	# Data
	embed_file: Optional[str] = None  # 'datset/train.txt'  # use pre-trained embeddings
	data_path: str = 'dataset/train.txt'
	val_data_path: Optional[str] = 'dataset/valid.txt'
	max_src_len: int = 400  # exclusive of special tokens such as EOS
	max_tgt_len: int = 100  # exclusive of special tokens such as EOS
	truncate_src: bool = False  # truncate to max_src_len? if false, drop example if too long
	truncate_tgt: bool = False  # truncate to max_tgt_len? if false, drop example if too long

	# Saving model automatically during training
	model_path_prefix: Optional[str] = model_dir + 'cached/seq2seq'
	keep_every_epoch: bool = False  # save all epochs, or only the best and the latest one?

	# Testing
	beam_size: int = 8
	min_out_len: int = 40
	max_out_len: Optional[int] = 100
	out_len_in_words: bool = False


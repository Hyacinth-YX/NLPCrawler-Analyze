ROOT_DIR = r"D:\git_of_skydownacai\\NLP_PROJECT\\"
data_path = ROOT_DIR + r"data\\"
tempfile = ROOT_DIR + r"temp\\"
model_path = ROOT_DIR + r"models\\"

#------------Corpus file------------------------------
CTB7_dir = "D:\\ctb\\7\\data\\"
CTB9_dir = "D:\\ctb\\9\\data\\"
CPB_dir = "D:\\cpb\\data\\"
CPB_verb_file = CPB_dir + "cpb3.0-verbs.txt"
CPB_nouns_file = CPB_dir + "cpb3.0-nouns.txt"

#------------Bert Model Related------------------------
bert_model_dir =ROOT_DIR +  r"bert_model\bert-chinese"
bert_vocab = bert_model_dir + "vocab.txt"
VOCAB_FILE = bert_model_dir + "\\vocab.txt"

#------------NER data source file path---------------
NER_source_data_dir = r"D:\nlpCorpus\CLUE\dataset\NER"
NER_TRAIN_SOURCE = NER_source_data_dir + r"\train.json"
NER_VALID_SOURCE = NER_source_data_dir + r"\dev.json"
NER_TEST_SOURCE  = NER_source_data_dir + r"\test.json"

#-------------NER data file path---------------------
NER_model_dir = model_path + r"BERT_NER\\"
NER_data_dir = data_path + r"NER\\"
NER_TRAIN = NER_data_dir + r"\train.json"
NER_VALID = NER_data_dir + r"\dev.json"
NER_TEST  = NER_data_dir + r"\test.json"
NER_labels = ['O', 'B-address', 'I-address', 'B-book', 'I-book', 'B-company', 'I-company', 'B-game', 'I-game', 'B-government', 'I-government', 'B-movie', 'I-movie', 'B-name', 'I-name', 'B-organization', 'I-organization', 'B-position', 'I-position', 'B-scene', 'I-scene']

#-------------NER data (People Daily) file path---------------------
NER_PEOPLEDAILY_dir = data_path + r"NER_PeopleDaily\\"
NER_PEOPLEDAILY_labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O"]
NER_PEOPLEDAILY_TRAIN =  NER_PEOPLEDAILY_dir+ r"train.json"
NER_PEOPLEDAILY_VALID =  NER_PEOPLEDAILY_dir+ r"dev.json"

#-------------SRL data file path-----------------------------------
SRL_predict_data_dir = data_path + r"SRL_Predict\\"
SRL_predict_label = ['O','I-norminalVerb', 'B-norminalVerb', 'I-verb', 'B-verb']
SRL_predict_train = SRL_predict_data_dir + "train.json"
SRL_predict_valid = SRL_predict_data_dir + "dev.json"
SRL_predict_model_dir = model_path + r"BERT_PredictLabeling\\"


SRL_data_dir = data_path + r"SRL_PA\\"
SRL_train = SRL_data_dir + "train.json"
SRL_valid = SRL_data_dir + "dev.json"
SRL_model_dir = model_path + r"BERT_SRL\\"
SRL_label = ['B-ARGM-QTY', 'B-ARG0-QTY', 'I-ARG0-MNR', 'I-ARG1-PRD', 'B-ARG0-PSE', 'B-rel-TPC', 'B-ARG2-PRD', 'I-rel-TPC', 'I-ARG3', 'I-rel-TMP', 'B-ARG0-PSR', 'I-ARG2-QTY', 'B-ARG1-FRQ', 'B-ARGM-DGR', 'B-ARG1-QTY', 'I-ARG1-TPC', 'I-ARG0', 'I-ARG0-PSE', 'I-ARG2-PSR', 'B-ARG0-CRD', 'B-rel-DIS', 'B-ARG1-DIS', 'B-ARG0', 'I-ARGM-TPC', 'I-ARGM-FRQ', 'I-ARG2-CRD', 'B-ARGM-DIR', 'I-ARG1', 'I-ARGM-DGR', 'I-ARG0-CND', 'I-ARGM-DIR', 'B-ARG2', 'B-ARG0-CND', 'B-ARG1-TPC', 'I-Sup', 'I-ARGM-T', 'I-ARGM-ADV', 'I-ARGM-CND', 'B-Sup', 'I-ARG1-DIS', 'B-ARG0-PRD', 'B-rel-MNR', 'B-ARGM', 'B-ARG2-CRD', 'B-ARGM-LOC', 'B-ARGM-EXT', 'I-ARG1-PSR', 'I-ARG0-ADV', 'I-ARGM-PRP', 'B-ARG1-CRD', 'I-ARG0-CRD', 'B-ARG0-MNR', 'B-rel-ADV', 'B-ARG2-QTY', 'I-ARGM-EXT', 'B-ARGM-T', 'I-ARG4', 'B-ARG2-PSR', 'I-ARGM-CRD', 'I-ARGM-PRD', 'B-ARGM-CND', 'I-ARGM-NEG', 'B-ARG1', 'B-ARG3', 'B-ARG0-ADV', 'O', 'I-ARGM-TMP', 'I-ARG1-QTY', 'B-ARGM-BNF', 'I-ARG1-CRD', 'I-ARG2-PRD', 'B-ARG1-PRD', 'I-rel', 'I-rel-MNR', 'I-ARG2-PSE', 'B-ARGM-MNR', 'B-ARGM-DIS', 'B-ARGM-FRQ', 'I-ARG0-PSR', 'I-ARGM-BNF', 'B-ARGM-NEG', 'I-ARGM-QTY', 'B-ARG2-PSE', 'I-ARGM-MNR', 'I-ARG2', 'I-ARGM-DIS', 'B-ARG1-PSR', 'I-ARG1-PSE', 'B-ARG1-PSE', 'I-ARG0-QTY', 'B-ARGM-CRD', 'I-ARGM-LOC', 'B-rel-TMP', 'B-rel-EXT', 'B-ARG3-TMP', 'B-ARGM-TMP', 'I-ARGM', 'B-ARGM-TPC', 'I-ARG1-FRQ', 'B-ARG4', 'I-ARG0-PRD', 'B-ARGM-PRD', 'B-ARGM-ADV', 'B-rel', 'I-rel-ADV', 'B-ARGM-PRP']
#---------------log file path-------------------------------------
log_path =r"D:\git_of_skydownacai\NLP_PROJECT\logs\\"



#---------------model training related--------------------------
device = "cuda:0"
gradient_accumulation_steps = 1
train_batch_size = 5
eval_batch_size = 5
learning_rate = 2e-5
num_train_epochs = 6
max_seq_length = 100
max_seq2_length = 10
warmup_proportion = 0.1
no_cuda = False
loss_scale = 0
fp16 = False

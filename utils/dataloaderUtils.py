import sys
sys.path.append("..")
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import config as args
from utils.loggerUtils import *
from utils.dataprocessUtils import convert_examples_to_features
logger = init_logger("bert_ner", logging_path=args.log_path)


def features_2_batch_iter(features,sampler = SequentialSampler,batch_size = 1):
    '''给定InputFeatures 列表,与指定的采样器，返回数据迭代器'''

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)
    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)
    sampler = sampler(data)
    #迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return iterator

def create_batch_iter(mode,processor,tokenizer,show_example = True):
    """构造数据加载迭代"""
    logger.info("加载%s迭代器" % mode)

    if mode == "train":
        examples = processor.get_train_examples()
        num_train_steps = int(
			len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        batch_size = args.train_batch_size
        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_dev_examples()
        batch_size = args.eval_batch_size

    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()

    # 特征
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, show_example,
											args.max_seq2_length)
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Num features = %d", len(features))
    logger.info("  Batch size = %d", batch_size)
    if mode == "train":
        sampler = RandomSampler
    elif mode == "dev":
        sampler = SequentialSampler
    else:
        raise ValueError("Invalid mode %s" % mode)

    iterator = features_2_batch_iter(features = features,sampler = sampler,batch_size = batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)



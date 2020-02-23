
'''
用于训练模型处理的一些公共库
@author : ljc
2020-02-20
'''
import os
import config.args as args
from utils.loggerUtils import init_logger
from pytorch_pretrained_bert.optimization import BertAdam
import torch
import numpy as np
from tqdm import tqdm
from utils.otherUtils import IO
import time
logger = init_logger("train", logging_path=args.log_path)
class Metrics:

	@staticmethod
	def BinarySeqLabel(pred : torch.tensor, truth : torch.tensor):

		pred = (pred > 0.5).int()

		TP = ((pred == 1) & (truth == 1)).cpu().sum().item()
		TN = ((pred == 0) & (truth == 0)).cpu().sum().item()
		FN = ((pred == 0) & (truth == 1)).cpu().sum().item()
		FP = ((pred == 1) & (truth == 0)).cpu().sum().item()

		p = 0 if TP == FP == 0 else TP / (TP + FP)

		r = 0 if TP == FN == 0 else TP / (TP + FN)

		F1 = 0 if p == r == 0 else 2 * r * p / (r + p)

		acc = (TP + TN) / (TP + TN + FP + FN)

		return (p, r, F1, acc)

	@staticmethod
	def MultiClassSeqLabel(pred: torch.tensor ,truth : torch.tensor):


		TP = ((pred == truth) * (pred > 0)).sum().item()

		TN = ((pred == truth) * (pred == 0)).sum().item()

		FN = ((pred != truth) * (pred > 0) * (truth != -1)).sum().item()

		FP = ((pred != truth) * (pred == 0)  * (truth != -1)).sum().item()


		p = 0 if TP == FP == 0 else TP / (TP + FP)

		r = 0 if TP == FN == 0 else TP / (TP + FN)

		F1 = 0 if p == r == 0 else 2 * r * p / (r + p)

		acc = (TP + TN) / (TP + TN + FP + FN)

		return (p, r, F1, acc)

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x

def save_model(model, output_dir,suffix):
	model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
	output_model_file = os.path.join(output_dir, "pytorch_model_%s.bin"%suffix)
	torch.save(model_to_save.state_dict(), output_model_file)

def load_model(model_dir):
	# Load a trained model that you have fine-tuned
	output_model_file = os.path.join(model_dir)
	model_state_dict = torch.load(output_model_file)
	return model_state_dict

class Trainner:
	'''在训练的'''
	def __init__(self,
				 model,train_dataloader,dev_dataloader,
				 loss,
				 cached_path : str ,
				 cached_point : int,
				 epoch_callback = None ,
				 batch_callback = None , ):
		'''
		:param model: 用于训练的模型
		:param loss: 模型的损失函数
		:param optimizer: 模型的优化器
		:param dataloader: 用于数据的加载
		:param cached_path: 模型保存路径
		:param cached_point:  每迭代该参数此
		:param epoch_callback: 每
		:param batch_callback:
		'''
		self.history = [] # 训练效果的历史数据
		self.loss = loss
		self.cached_path = cached_path # 保存模型的路径
		self.cached_point = cached_point
		self.model = model
		self.train_dataloader = train_dataloader
		self.dev_dataloader = dev_dataloader
		self.epoch_callback = epoch_callback
		self.batch_callback = batch_callback
		logger.info("initialize trainner success")

class SeqLabelingTrainer(Trainner):

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def fit(self,num_epoch,num_train_steps, verbose=1):

		logger.info("Start Training...")
		# ------------------判断CUDA模式----------------------
		device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
		# ---------------------优化器-------------------------
		param_optimizer = list(self.model.named_parameters())

		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		t_total = num_train_steps


		## ---------------------GPU半精度fp16-----------------------------
		if args.fp16:
			try:
				from apex.optimizers import FP16_Optimizer
				from apex.optimizers import FusedAdam
			except ImportError:
				raise ImportError(
					"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

			optimizer = FusedAdam(optimizer_grouped_parameters,
								  lr=args.learning_rate,
								  bias_correction=False,
								  max_grad_norm=1.0)
			if args.loss_scale == 0:
				optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
			else:
				optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
		## ------------------------GPU单精度fp32---------------------------
		else:
			optimizer = BertAdam(optimizer_grouped_parameters,
								 lr=args.learning_rate,
								 warmup=-1,
								 t_total=-1)
		# ---------------------模型初始化----------------------
		if args.fp16:
			self.model.half()

		self.model.to(device)

		T_losses, T_p, T_r, T_f1, T_acc = [], [], [], [], []
		V_losses, V_p, V_r, V_f1, V_acc = [], [], [], [], []
		history = {
			"train_loss": T_losses,
			"train_precision": T_p,
			"train_recall": T_r,
			"train_f1": T_f1,
			"train_acc":T_acc,
			"dev_loss": V_losses,
			"dev_precision": V_p,
			"dev_recall": V_r,
			"dev_f1": V_f1,
			"dev_acc": V_acc
		}
		# ------------------------训练------------------------------
		best_f1 = 0
		start = time.time()
		global_step = 0
		for e in range(num_epoch):
			losses, p,r, f1, acc = [], [], [], [], []
			pbar = tqdm(total=len(self.train_dataloader))
			for step, batch in enumerate(self.train_dataloader):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids, output_mask = batch
				# print("input_id", input_ids)
				# print("input_mask", input_mask)
				# print("segment_id", segment_ids)
				b_prob = self.model(input_ids, segment_ids, input_mask)
				shape = list(b_prob.size())
				#修改shape 方便计算
				b_predict = b_prob.argmax(dim = 2).view(shape[0] * shape[1])
				b_y = label_ids.view(shape[0] * shape[1])
				b_prob = b_prob.view(shape[0] * shape[1], shape[2])
				train_loss = self.loss(b_prob, b_y)

				if args.gradient_accumulation_steps > 1:
					train_loss = train_loss / args.gradient_accumulation_steps

				if args.fp16:
					optimizer.backward(train_loss)
				else:
					train_loss.backward()

				b_p, b_r, b_f1, b_acc = Metrics.MultiClassSeqLabel(b_predict, b_y)
				p.append(b_p)
				r.append(b_r)
				f1.append(b_f1)
				acc.append(b_acc)
				losses.append(train_loss.item())

				if (step + 1) % args.gradient_accumulation_steps == 0:
					# modify learning rate with special warm up BERT uses
					lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr_this_step
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1

				pbar.set_description(
					'Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f| PRECISION: %1.3f | RECALL: %1.3f' % (e, train_loss.item(), b_f1, b_acc,b_p,b_r))

				pbar.update(1)
			losses, p, r, f1, acc = list(map(np.mean,[losses, p, r, f1, acc]))
			pbar.clear()
			pbar.close()
			logger.info('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
						(e, losses, f1, acc,p, r))
			T_p.append(p)
			T_r.append(r)
			T_f1.append(f1)
			T_acc.append(acc)
			T_losses.append(losses)

		# -----------------------验证----------------------------

			losses, p,r, f1, acc = [], [], [], [], []

			with torch.no_grad():

				for step, batch in enumerate(self.dev_dataloader):

					batch = tuple(t.to(device) for t in batch)

					input_ids, input_mask, segment_ids, label_ids, output_mask = batch
					# print("input_id", input_ids)
					# print("input_mask", input_mask)
					# print("segment_id", segment_ids)
					b_prob = self.model(input_ids, segment_ids, input_mask)
					shape = list(b_prob.size())
					# 修改shape 方便计算
					b_predict = b_prob.argmax(dim=2).view(shape[0] * shape[1])
					b_y = label_ids.view(shape[0] * shape[1])
					b_prob = b_prob.view(shape[0] * shape[1], shape[2])
					dev_loss = self.loss(b_prob, b_y)
					b_p, b_r, b_f1, b_acc = Metrics.MultiClassSeqLabel(b_predict, b_y)
					p.append(b_p)
					r.append(b_r)
					f1.append(b_f1)
					acc.append(b_acc)
					losses.append(dev_loss.item())

			losses, p, r, f1, acc = list(map(np.mean,[losses, p, r, f1, acc]))
			logger.info('Dev Set   | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
						(losses, f1, acc,p, r))

			V_p.append(p)
			V_r.append(r)
			V_f1.append(f1)
			V_acc.append(acc)
			V_losses.append(losses)


			if e % self.cached_point == 0 and e > 0:

				IO.saveJson(self.cached_path + "History_epoch.json"  ,history)

				save_model(self.model,self.cached_path,str(e))

				logger.info("Model and History Saved ! ")

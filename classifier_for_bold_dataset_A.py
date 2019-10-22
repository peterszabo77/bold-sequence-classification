# Bold sequence data classifier

# IMPORTS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import os
import matplotlib
import matplotlib.pyplot as plt

DATADIR = 'bolddata'
MODELSAVEDIR = 'model'
DATAFILE = 'bold_data.csv'
N_EPOCHS = 20
N_GENERA = 10
BATCHSIZE = N_GENERA*10
RANDOMSEED = 1
LEARNING_RATE = 0.001
TRAIN_PROP = 0.8
EVAL_PROP = 1-TRAIN_PROP

# LSTMNetwork parameters
N_HIDDEN = 64

nucldict = {'-':0,'A':1,'T':2,'C':3,'G':4}

def onehot(idx,length):
	result = np.zeros(length)
	result[idx]=1
	return result

def seq_to_vector(seq):
	onehotlist=[]
	for nucl in seq:
		onehotlist.append(onehot(nucldict[nucl],len(nucldict)))
	result = np.stack(onehotlist)
	return result

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def gettarget(batch,pred):
	m1 = batch[:,[0],:].expand_as(batch) # tensor with all-1 channels, where nucleotide is '-'
	return torch.where(m1==1, pred, batch)

def savemodel(model):
	try:
		os.mkdir(MODELSAVEDIR)
	except:
		pass
	filepath = os.path.join(MODELSAVEDIR,'model.pt')
	torch.save(model.state_dict(), filepath)

def loadmodel():
	filepath = os.path.join(MODELSAVEDIR,'model.pt')
	try:
		model = LSTMNetwork().double()
		model.load_state_dict(torch.load(filepath))
		model.eval()
		return model
	except:
		return None

# DATASET
class BoldDataset():
	def __init__(self):
		filepath = os.path.join(DATADIR, DATAFILE)
		data = pd.read_csv(filepath, sep='\t')
		N_allrecords = data.shape[0]
		# filtering records of the most frequent genera
		print('filtering records of the '+str(N_GENERA)+' most frequent genera:')
		genus_counts = data['genus'].value_counts().sort_values(ascending=False).iloc[0:N_GENERA]
		self.genus_names = list(genus_counts.index.sort_values())
		self.genus_to_idx = {e:i for i, e in enumerate(self.genus_names)}
		self.idx_to_genus = {i:e for i, e in enumerate(self.genus_names)}
		print(genus_counts)
		data = data[data['genus'].isin(genus_counts.index)]
		data.sort_values('genus',inplace=True)
		self.seq_length = len(data['sequence'].iloc[0])

		# separate to train and evaluation dataset (by genuses)
		## train_sizes per genus
		self.train_counts_by_genus = {g:int(c*TRAIN_PROP) for g, c in genus_counts.to_dict().items()}

		databygenus = {g:data[data['genus'] == g].drop('genus',axis=1) for g in self.genus_names}
		self.train_data_by_genus = {g:seqs[0:self.train_counts_by_genus[g]] for g, seqs in databygenus.items()}
		self.eval_data_by_genus = {g:seqs[self.train_counts_by_genus[g]:] for g, seqs in databygenus.items()}

		# batch generation utilities
		self.N_records = data.shape[0]
		self.N_trainrecords = sum([v.shape[0] for k,v in self.train_data_by_genus.items()])
		self.N_evalrecords = sum([v.shape[0] for k,v in self.eval_data_by_genus.items()])
		print('\nWhole dataset: {} records'.format(self.N_records))
		print('Train dataset: {} records'.format(self.N_trainrecords))
		print('Evaluation dataset: {} records'.format(self.N_evalrecords))
		self.N_batches = int(self.N_trainrecords / BATCHSIZE)
		self.genusbatchsize = int(BATCHSIZE / N_GENERA)
		if self.genusbatchsize < 1:
			print('Error: genus batch size is zero. Increase batch size!')

	def get_batch(self, batchsize, device):
		# returns an array of shape (batchsize, seqlength, features)
		labels = []
		sequences = []
		for g in self.genus_names:
			genussample = self.train_data_by_genus[g].sample(self.genusbatchsize, replace = True)
			for label, sequence in genussample.iterrows():
				sequences.append(seq_to_vector(sequence[0]))
				labels.append(g)
		batchtensor = torch.from_numpy(np.stack(sequences))
		labelstensor = torch.tensor([self.genus_to_idx[e] for e in labels])
		return batchtensor.to(device), labelstensor.to(device)

	def get_evalbatch(self, batchsize, device):
		# returns an array of shape (batchsize, seqlength, features)
		labels = []
		sequences = []
		for g in self.genus_names:
			genussample = self.eval_data_by_genus[g].sample(self.genusbatchsize, replace = True)
			for label, sequence in genussample.iterrows():
				sequences.append(seq_to_vector(sequence[0]))
				labels.append(g)
		batchtensor = torch.from_numpy(np.stack(sequences))
		labelstensor = torch.tensor([self.genus_to_idx[e] for e in labels])
		return batchtensor.to(device), labelstensor.to(device)

# NETWORK

class LSTMNetwork(nn.Module):
	def __init__(self):
		super(LSTMNetwork,self).__init__()

		self.lstm = nn.LSTM(input_size=len(nucldict), hidden_size=N_HIDDEN, batch_first=True)
		self.fc1 = nn.Linear(in_features=N_HIDDEN, out_features=N_HIDDEN)
		self.fc2 = nn.Linear(in_features=N_HIDDEN, out_features=N_GENERA)

	def forward(self,x, apply_softmax=False):
		# input shape: (batchsize, seq_len, input_size)
		# output (out, (h_n,c_n)) shape :
		# 	out: (batchsize, seq_len, num_directions * hidden_size) contains the output features (h_t) from the last layer of the LSTM, for each t
		# 	hn: (batchsize, num_layers * num_directions, hidden_size) contains the hidden state for t = seq_len
		# 	cn: (batchsize, num_layers * num_directions, hidden_size) contains the cell state for t = seq_len
		# output shape: (batchsize, num_of_classes)

		out, (h_n, c_n) = self.lstm(x)
		h_n = h_n.view(BATCHSIZE,-1) # (batchsize, num_layers * num_directions, hidden_size) -> (batchsize, hidden_size)
		fc1_out = F.leaky_relu(self.fc1(h_n))
		fc2_out = self.fc2(fc1_out)

		if apply_softmax:
			y = F.softmax(fc2_out, dim=1)
		else:
			y = fc2_out
		return y

def get_accuracy(pred, target_idxs):
	pred_idxs = pred.max(dim=1)[1]
	accuracy = torch.eq(pred_idxs, target_idxs).sum().item()/pred.shape[0]
	return accuracy

def savefigure(traindata):
	fig = plt.figure(figsize=(8, 8))
	ax1 = fig.add_subplot(2,1,1)
	ax1.plot(traindata['epoch'], traindata['train_loss'], label='train')
	ax1.plot(traindata['epoch'], traindata['eval_loss'], label='evaluation')
	plt.ylabel('Loss')
	plt.legend(loc='lower left')
	ax2 = fig.add_subplot(2,1,2)
	ax2.plot(traindata['epoch'], traindata['train_acc'], label='train')
	ax2.plot(traindata['epoch'], traindata['eval_acc'], label='evaluation')
	ax2.plot([1,N_EPOCHS],[1/N_GENERA,1/N_GENERA], color='red', label='reference (random guess)')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim(0, 1.0)
	plt.legend(loc='upper left')
	ax1.figure.savefig('training.pdf')

bolddataset = BoldDataset()

cuda = torch.cuda.is_available()
mydevice = torch.device("cuda" if cuda else "cpu")
print("\nUsing CUDA: {}\n".format(cuda))

set_seed_everywhere(RANDOMSEED, cuda)

mynetwork = LSTMNetwork().double()
mynetwork = mynetwork.to(mydevice)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(mynetwork.parameters(), lr=LEARNING_RATE)

batch, labels = bolddataset.get_batch(BATCHSIZE, mydevice)

print('training...')

trainingprocess = {'epoch':[], 'train_loss':[], 'train_acc':[], 'eval_loss':[], 'eval_acc':[]}
maxaccuracy = 0

for epoch_idx in range(N_EPOCHS):
	mean_train_loss = 0
	mean_train_accuracy = 0
	
	mynetwork.train()
	for batch_idx in range(bolddataset.N_batches):
		batch, label_idxs = bolddataset.get_batch(BATCHSIZE, mydevice)
		# the training routine is these 5 steps:
		# --------------------------------------
		# step 1. zero the gradients
		optimizer.zero_grad()
		# step 2. compute the output
		pred = mynetwork(batch)
		# step 3. compute the loss
		loss = loss_func(pred, label_idxs)
		
		#statistics
		acc = get_accuracy(pred, label_idxs)
		mean_train_loss = (mean_train_loss*(batch_idx)+loss.item())/(batch_idx+1)
		mean_train_accuracy = (mean_train_accuracy*(batch_idx)+acc)/(batch_idx+1)
		
		# step 4. use loss to produce gradients
		loss.backward()
		# step 5. use optimizer to take gradient step
		optimizer.step()

	mean_eval_loss = 0
	mean_eval_accuracy = 0
	mynetwork.eval()
	for batch_idx in range(bolddataset.N_batches):
		batch, label_idxs = bolddataset.get_evalbatch(BATCHSIZE, mydevice)
		# the training routine is these 5 steps:
		# --------------------------------------
		# step 1. compute the output
		pred = mynetwork(batch)
		# step 2. compute the loss
		loss = loss_func(pred, label_idxs)
		
		#statistics
		acc = get_accuracy(pred, label_idxs)
		mean_eval_loss = (mean_eval_loss*(batch_idx)+loss.item())/(batch_idx+1)
		mean_eval_accuracy = (mean_eval_accuracy*(batch_idx)+acc)/(batch_idx+1)

	trainingprocess['epoch'].append(epoch_idx+1)
	trainingprocess['train_loss'].append(mean_train_loss)
	trainingprocess['eval_loss'].append(mean_eval_loss)
	trainingprocess['train_acc'].append(mean_train_accuracy)
	trainingprocess['eval_acc'].append(mean_eval_accuracy)
	
	print('epoch [{}/{}] - loss: {:.4f}, eloss: {:.4f}, acc: {:.4f}, eacc: {:.4f}'.format(epoch_idx+1, N_EPOCHS, mean_train_loss, mean_eval_loss, mean_train_accuracy, mean_eval_accuracy))
	
	if mean_eval_accuracy > maxaccuracy:
		maxaccuracy = mean_eval_accuracy
		savemodel(mynetwork)

savefigure(trainingprocess)

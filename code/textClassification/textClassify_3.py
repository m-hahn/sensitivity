
import random
import sys

objectiveName = "graphParser"

print(sys.argv)




import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--dataset", type=str, default = "subj")


args=parser.parse_args()



prescribedID = None

if prescribedID is not None and prescribedID != "NONE":
  myID = int(prescribedID)
else:
  myID = random.randint(0,10000000)




posUni = set() 
posFine = set() 


from math import log, exp, sqrt
from random import random, shuffle, randint
import os



originalDistanceWeights = {}



import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random
import numpy as np


import dataloader

args.path = "/u/scr/mhahn/software/sent-conv-torch/data"

# https://github.com/asappresearch/sru/blob/master/classification/train_classifier.py
if args.dataset == 'mr':
    data, label = dataloader.read_MR(args.path)
elif args.dataset == 'subj':
    data, label = dataloader.read_SUBJ(args.path)
elif args.dataset == 'cr':
    data, label = dataloader.read_CR(args.path)
elif args.dataset == 'mpqa':
    data, label = dataloader.read_MPQA(args.path)
elif args.dataset == 'trec':
    train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path)
    data = train_x + test_x
    label = None
elif args.dataset == 'sst':
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
    data = train_x + valid_x + test_x
    label = None
else:
    raise Exception("unknown dataset: {}".format(args.dataset))

words = {}
for sent in data:
   for word in sent:
      words[word] = words.get(word, 0) + 1
words = sorted(list(words.items()), key=lambda x:-x[1])
itos = [x[0] for x in words]
stoi = dict(list(zip(itos, list(range(len(itos))))))



import os



word_embedding_size = 50
dropout_rate = 0.1
rnn_dim = 100
lstm_layers = 1
bilinearSize = 20
labelMLPDimension = 20
lr_lm = 0.01
vocab_size = 10000
input_dropoutRate = 0.1
tokensPerBatch =10

def toDevice(x):
   if True:
     return x
   else:
     return x.cuda()



from torch import optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WINDOW = 5
import scipy.special
numberOfFeatures = int(scipy.special.binom(2*WINDOW, 3))
print("FEATURES", numberOfFeatures)

class Model(nn.Module):
   def __init__(self):
       super(Model, self).__init__()
       self.words_embeddings = torch.nn.Embedding(num_embeddings = len(itos)+3, embedding_dim = 50)
       self.position_embeddings = torch.nn.Embedding(num_embeddings = 30, embedding_dim = 50)

       self.dropout = nn.Dropout(dropout_rate)
       self.inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)
       self.softmax_layer = torch.nn.Softmax()
       self.logsoftmax = torch.nn.LogSoftmax()
       self.logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)

       self.first_layer = torch.nn.Linear(350, 400)
       self.relu = torch.nn.ReLU()
       self.output = torch.nn.Linear(400, 2)

       self.weights = torch.nn.Linear(1, numberOfFeatures)
       self.loss = torch.nn.CrossEntropyLoss(reduce=None, reduction="none")
       self.optimizer = optim.Adam(self.parameters(), lr = lr_lm)

   def forward(self, current, printHere=False, computeAccuracy=False, doDropout=True):
       global biasHead
       global crossEntropy
       global batchSize
       batchSize = 1
  
       current, targets = current 
       lengths = list(map(len, current))
       maxLength = max(lengths)
       input_words = []
       input_words = []
     #  print(targets, current)
       for i in range(maxLength):
          input_words.append([encodeWord(x[i]) for x in current])

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       self.optimizer.zero_grad()

       input_words_padded = [[2] for _ in range(WINDOW)] + input_words + [[2] for _ in range(WINDOW)]
       words_layer = self.words_embeddings(Variable(torch.LongTensor(input_words_padded).to(device).transpose(0,1))).squeeze(0)
#       print(words_layer.size())
       relevantPositions = Variable(torch.LongTensor([[x,y,z] for x in range(0,2*WINDOW) for y in range(0, 2*WINDOW) for z in range(0, 2*WINDOW) if x < y and y < z]).to(device))
     #  print(relevantPositions.size(), numberOfFeatures)
      # quit()

     #  print(relevantPositions.size())
       position_layer = self.position_embeddings(relevantPositions) # number of pairs * 2 * 50
    #   print(position_layer.size())

       # for each pair of words, extract a feature
      
       # TODO somehow have to pad the embeddings
       relativeIndicesMatrix = torch.stack([x+relevantPositions-WINDOW for x in range(len(input_words))], dim=0) # sentLength * number_of_pairs * 3
   #    print(relativeIndicesMatrix.size())
  
       context_word_embeddings = words_layer[WINDOW + 1 + relativeIndicesMatrix] # sentLength * numberOfPairs * 3 * 50
  #     print(context_word_embeddings.size())

       centralWord = words_layer[WINDOW:-WINDOW].unsqueeze(1) # sentLength * 1 * 50
 #      print(centralWord.size())
       context_word_embeddings_with_pos = torch.cat([context_word_embeddings.view(-1, numberOfFeatures, 150), position_layer.view(numberOfFeatures, 150).unsqueeze(0).expand(context_word_embeddings.size()[0], -1, -1), centralWord.expand(-1, numberOfFeatures, -1)], dim=2)
#       print(context_word_embeddings_with_pos.size()) # sentLength * 190 * 250

       intermediate = self.first_layer(context_word_embeddings_with_pos)
       nonlinear = self.relu(intermediate)
       features = self.output(nonlinear)

    #   print(features.size(), self.weights.weight.size())
       #print(features.size())
       weights = (self.weights.weight * features).mean(dim=1).mean(dim=0)
       target = torch.LongTensor(targets).to(device)
   #    print(target.size(), target.max())
  #     print(weights.size())
       #print(weights.size(), target.size())
       loss = self.loss(weights.view(1,2), target)
       if printHere:
#           print(self.weights.weight)
           print("Relation identification loss", loss.mean())
#       print(words_layer.size(), position_layer.size())
#       quit()

       return loss.sum(), len(targets)

   def backward(self, loss):
       # regularization
       WEIGHTS_L2 = 1e-5
       loss = loss + WEIGHTS_L2 * self.weights.weight.pow(2).mean()
       loss.backward()
       self.optimizer.step()





model = Model().to(device)




def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)


import torch.nn.functional






devLosses = []

import sys

counter = 0
epochs = 0
import math
crossEntropy = math.log(2)
while True:
  epochs += 1
  labeledData = list(zip(data, label))
  shuffle(labeledData)
  for partition in labeledData:

       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum = model.forward(([partition[0]], [partition[1]]), printHere)
       decay = 0.9999 ** wordNum
       crossEntropy = (1-decay) * float(loss)/wordNum + decay * crossEntropy
       if wordNum == 0:
          assert loss is 0
       else:
          model.backward(loss)
       if printHere:
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devLosses)
           print("Running Average ", crossEntropy)











 

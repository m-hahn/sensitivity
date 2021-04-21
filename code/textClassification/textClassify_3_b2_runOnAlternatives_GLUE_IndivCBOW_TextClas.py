# Works well

import random
import sys
import re

objectiveName = "graphParser"

print(sys.argv)




import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--dataset", type=str, default = "MNLI")
parser.add_argument("--lr_lm", type=float, default = 0.001)


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

devName = "dev_matched" if args.dataset == "MNLI" else "dev"

with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/processed/{devName}.raw.input0", "r") as inFile0:
   data_dev0 = inFile0.read().strip().split("\n")
with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/processed/{devName}.label", "r") as inFile0:
   label_dev = [x for x in inFile0.read().strip().split("\n")]


with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/processed/train.raw.input0", "r") as inFile0:
   data_train0 = inFile0.read().strip().split("\n")
with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/processed/train.label", "r") as inFile0:
   label_train = [x for x in inFile0.read().strip().split("\n")]


itos_labels = sorted(list(set(label_dev)))
stoi_labels = dict(list(zip(itos_labels, range(len(itos_labels)))))

label_dev = [stoi_labels[x] for x in label_dev]
label_train = [stoi_labels[x] for x in label_train]

print(len(label_dev), len(label_train))
#quit()

# This function is from dataloader.py
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



if True:
  data_train = [clean_str(x).split(" ") for x in (data_train0)]
  data_dev = [clean_str(x).split(" ") for x in (data_dev0)]



words = {}
for sent in data_train:
   for word in sent:
      words[word] = words.get(word, 0) + 1
words = sorted(list(words.items()), key=lambda x:-x[1])
itos = [x[0] for x in words]
stoi = dict(list(zip(itos, list(range(len(itos))))))



import os



word_embedding_size = 50
dropout_rate = 0.2
rnn_dim = 100
lstm_layers = 1
bilinearSize = 20
labelMLPDimension = 20
vocab_size = 50000
input_dropoutRate = 0.1
tokensPerBatch =10
batchSize=32

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
       self.words_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 300)
       with open("/u/scr/mhahn/glove/glove.840B.300d.txt", "r") as inFile:
          counter = 0 
          for line in inFile:
             counter += 1
             if counter > 50000:
                 break
             if len(line) < 10:
                 continue
             line = line.split(" ")
             if line[0] in stoi and stoi[line[0]] < vocab_size:
                 if len(line) != 301:
                    print("ERROR", line[:5])
                    continue
                 self.words_embeddings.weight.data[stoi[line[0]]+3] = torch.FloatTensor([float(x) for x in line[1:]])
  

       self.dropout = nn.Dropout(dropout_rate)
       self.inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)
       self.softmax_layer = torch.nn.Softmax()
       self.logsoftmax = torch.nn.LogSoftmax()
       self.logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)

       #self.first_layer = torch.nn.Linear(350, 400)
       self.relu = torch.nn.ReLU()
       self.output = torch.nn.Linear(256, len(itos_labels))
       self.hidden = torch.nn.Linear(300, 256)

       self.weights = torch.nn.Linear(1, numberOfFeatures)
       self.loss = torch.nn.CrossEntropyLoss(reduce=None, reduction="none")
       self.optimizer = optim.Adam(self.parameters(), lr = args.lr_lm)

#       self.rnn1 = nn.LSTM(300, 128, 1, bidirectional=True)
 #      self.rnn2 = nn.LSTM(300, 128, 1, bidirectional=True)

   def forward(self, current, printHere=False, computeAccuracy=False, train=True, getPredictions=False):
       global biasHead
       global crossEntropy
       if getPredictions:
          print("####")
          print(current[0])
       targets = [x[1][1] for x in current]
       current = [x[1][0] for x in current]
       premises = [x for x in current]
       lengths_premises = list(map(len, premises))
       maxLength_premises = max(lengths_premises)
       input_words_premises = [[encodeWord(x) for x in y] + [2 for _ in range(maxLength_premises-len(y))] for y in premises]

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       self.optimizer.zero_grad()

       words_layer_premise = self.words_embeddings(Variable(torch.LongTensor(input_words_premises).to(device))).detach()
       if train:
         words_layer_premise = self.dropout(words_layer_premise)
#       representation_premise, _ = self.rnn1(words_layer_premise.transpose(0,1))
       #print(representation_premise.size())
       representation_premise = words_layer_premise.mean(dim=1)
       convolved = representation_premise
       convolved= torch.tanh(self.hidden(convolved))
       predicted = self.output(convolved)
       target = torch.LongTensor(targets).to(device)
       loss = self.loss(predicted, target)
       if printHere:
           print("Relation identification loss", float(loss.mean()))

       prediction = torch.argmax(predicted, dim=1) #torch.where(predicted[:,0] < predicted[:,1], 1+0*target, 0*target)
#       print(prediction)

       if getPredictions:
         
         probs = torch.softmax(predicted, dim=1)
 #        print(probs[:,1]-probs[:,0])
#         quit()
         return probs
#         return 2*prediction.float()-1 # for binary prediction
       else:
         return loss.sum(), batchSize, float((prediction == target).float().mean())

   def backward(self, loss):
       # regularization
       WEIGHTS_L2 = 1e-5
#       loss = loss + WEIGHTS_L2 * self.weights.weight.pow(2).mean()
       loss.backward()
       self.optimizer.step()





model = Model().cuda() #to(device)




def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 0.69
accuracyMean = 0.5

def encodeWord(w):
   return stoi[w]+3 if w in stoi and stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)


import torch.nn.functional

devAccuracies = []

import sys
crossEntropyDev = 0.69
accuracyMeanDev = 0.5

counter = 0
epochs = 0
import math
crossEntropy = math.log(2)
lastDevAcc = 0
while epochs < 50:
  epochs += 1
  
  counter = 0
  labeledData = list(zip(data_train, label_train))
  shuffle(labeledData)
  print("DATAPOINTS", len(labeledData))
  labeledData = enumerate(labeledData)
  try:
    while True:
       batch = [next(labeledData) for _ in range(batchSize)]
#       print(batch)
       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum, accuracy = model.forward(batch, printHere)
       decay = 0.9999 ** wordNum
       crossEntropy = (1-decay) * float(loss)/wordNum + decay * crossEntropy
       accuracyMean = (1-decay) * float(accuracy) + decay * accuracyMean 

       if wordNum == 0:
          assert loss is 0
       else:
          model.backward(loss)
       if printHere:
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devAccuracies)
           print(args)
           print("Running Average ", crossEntropy, accuracyMean)
           print("DEV",  "CE", crossEntropyDev,  "ACC", accuracyMeanDev, "Epochs", epochs)
  
  except StopIteration:
    print("Next Epoch", counter*batchSize, epochs) 
  labeledData = list(zip(data_dev, label_dev))
  shuffle(labeledData)
  print("DATAPOINTS", len(labeledData))
  labeledData = enumerate(labeledData)
  crossEntropyDev = 0
  accuracyMeanDev = 0
  overallIndices = 0
  try:
    while True:
       batch = [next(labeledData) for _ in range(batchSize)]
       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum, accuracy = model.forward(batch, printHere, train=False)
#       print("DEV", float(loss)/batchSize, float(accuracy))
       decay = 0.9999 ** wordNum
       crossEntropyDev += float(loss)/wordNum
       accuracyMeanDev += float(accuracy)
       overallIndices += 1
       if printHere:
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devAccuracies)
           print(args)
           print("Running Average ", crossEntropy, accuracyMean)
  except StopIteration:
    print("Next Epoch", counter*batchSize, epochs) 
  crossEntropyDev /= overallIndices
  accuracyMeanDev /= overallIndices
  devAccuracies.append(accuracyMeanDev)
  if accuracyMeanDev > lastDevAcc:
     lastDevAcc = accuracyMeanDev
  else:
     print("Not improving any more")
     break


def counts(x):
   counts = {y : 0 for y in itos_labels}
   for z in x:
      counts[itos_labels[z]] += 1/len(x)
   return counts
print(counts(label_dev))

print("DEV ACCURACY", lastDevAcc)


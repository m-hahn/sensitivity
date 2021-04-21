# Works well

import random
import sys

objectiveName = "graphParser"

print(sys.argv)




import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--dataset", type=str, default = "SST-2")


args=parser.parse_args()


assert args.dataset == "SST-2"

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


with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/train.tsv", "r") as inFile:
   data = [x.split("\t") for x in inFile.read().strip().split("\n")[1:]]
data_train, label_train = list(zip(*data))
label_train = [int(x) for x in label_train]
data_train = [x.strip().split(" ") for x in data_train]
with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev.tsv", "r") as inFile:
   data_dev, label_dev = zip(*[x.split("\t") for x in inFile.read().strip().split("\n")[1:]])
label_dev = [int(x) for x in label_dev]
data_dev = [x.strip().split(" ") for x in data_dev]

words = {}
for sent in data_train + data_dev:
   for word in sent:
      words[word] = words.get(word, 0) + 1
words = sorted(list(words.items()), key=lambda x:-x[1])
itos = [x[0] for x in words]
stoi = dict(list(zip(itos, list(range(len(itos))))))


import os



word_embedding_size = 50
dropout_rate = 0.4
rnn_dim = 100
lstm_layers = 1
bilinearSize = 20
labelMLPDimension = 20
lr_lm = 0.01
vocab_size = 10000
input_dropoutRate = 0.2
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
       self.words_embeddings = torch.nn.Embedding(num_embeddings = len(itos)+3, embedding_dim = 400)
       #self.position_embeddings = torch.nn.Embedding(num_embeddings = 30, embedding_dim = 50)

       self.dropout = nn.Dropout(dropout_rate)
       self.inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)
       self.softmax_layer = torch.nn.Softmax()
       self.logsoftmax = torch.nn.LogSoftmax()
       self.logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)

       #self.first_layer = torch.nn.Linear(350, 400)
       self.relu = torch.nn.ReLU()
       self.output = torch.nn.Linear(256, 2)

       self.weights = torch.nn.Linear(1, numberOfFeatures)
       self.loss = torch.nn.CrossEntropyLoss(reduce=None, reduction="none")
       self.optimizer = optim.Adam(self.parameters(), lr = lr_lm)

       self.rnn = nn.LSTM(400, 128, 1, bidirectional=True) 

   def forward(self, current, printHere=False, computeAccuracy=False, train=True, getPredictions=False, outFile=None):
       global biasHead
       global crossEntropy
       if getPredictions:
          print("####")
          print(current[0])
       targets = [x[1][1] for x in current]
       current = [x[1][0] for x in current]
       lengths = list(map(len, current))
       maxLength = max(lengths)
       input_words = []
       input_words = []
       #print(targets, current)
       if getPredictions:
          print("====")
          print(current[0])
       input_words = [[encodeWord(x) for x in y] + [2 for _ in range(maxLength-len(y))] for y in current]

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       self.optimizer.zero_grad()

       input_words_padded = input_words
       words_layer = self.words_embeddings(Variable(torch.LongTensor(input_words_padded).to(device)))
       if train:
          words_layer = self.dropout(words_layer)
       #print(words_layer.size())
       words_layer = words_layer.transpose(1,0)
       _, convolved = self.rnn(words_layer)
       convolved = convolved[0].transpose(0,1).contiguous().view(len(current), 256)
       #print(convolved.size())
       #quit()
       #convolved= self.relu(convolved)
       if train:
         convolved = self.dropout(convolved)
       pooled = convolved
       predicted = self.output(pooled)
       target = torch.LongTensor(targets).to(device)
       loss = self.loss(predicted, target)
       if printHere:
           print("Relation identification loss", float(loss.mean()))

       prediction = torch.where(predicted[:,0] < predicted[:,1], 1+0*target, 0*target)
       if outFile is not None:
         probs = torch.softmax(predicted, dim=1).log()
#         print(probs)
         for i in range(len(current)):
             print("\t".join([" ".join(current[i]), str(float(probs[i,0]))]), file=outFile)
       if getPredictions:
         
         probs = torch.softmax(predicted, dim=1)
 #        print(probs[:,1]-probs[:,0])
#         quit()
         return probs[:,1] - probs[:,0] # for continuous prediction
#         return 2*prediction.float()-1 # for binary prediction
       else:
         return loss.sum(), batchSize, float((prediction == target).float().mean())

   def backward(self, loss):
       # regularization
       WEIGHTS_L2 = 1e-5
#       loss = loss + WEIGHTS_L2 * self.weights.weight.pow(2).mean()
       loss.backward()
       self.optimizer.step()





model = Model().to(device)




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

DEV_END = int(len(data)/10)
TEST_END = 2*int(len(data)/10)


devLosses = []

import sys
crossEntropyDev = 0.69
accuracyMeanDev = 0.5

counter = 0
epochs = 0
import math
crossEntropy = math.log(2)
lastDevAcc = 0
while True:
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
           print(devLosses)
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
  outFile = None
  try:
    while True:
       batch = [next(labeledData) for _ in range(batchSize)]
       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum, accuracy = model.forward(batch, printHere, train=False, outFile=outFile)
#       print("DEV", float(loss)/batchSize, float(accuracy))
       decay = 0.9999 ** wordNum
       crossEntropyDev += float(loss)/wordNum
       accuracyMeanDev += float(accuracy)
       overallIndices += 1
       if printHere:
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devLosses)
           print("Running Average ", crossEntropy, accuracyMean)
  except StopIteration:
    print("Next Epoch", counter*batchSize, epochs) 
  crossEntropyDev /= overallIndices
  accuracyMeanDev /= overallIndices

  if accuracyMeanDev > lastDevAcc:
     lastDevAcc = accuracyMeanDev
  else:
     break



with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_predictions_finetuned.tsv', "r") as inFile:
   with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_LSTMpredictions_finetuned.tsv', "w") as outFile:
     print("\t".join(['Sentence', 'Predicted']), file=outFile)
     for line in inFile:
       sentence, _, _ = line.strip().split("\t")
       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum, accuracy = model.forward([(None, (sentence.strip().split(" "), 1))], printHere, train=False, outFile=outFile)
       decay = 0.9999 ** wordNum
       overallIndices += 1
       if printHere:
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devLosses)
           print("Running Average ", crossEntropy, accuracyMean)


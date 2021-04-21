# Mofified from textClassify_3_b1.py

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


p_Recursion = 0.55
p_Negation = 0.0
def constructFormulaBrackets():
   choice = random()
   if choice < p_Recursion/2:
      return ["T"], 1
   elif choice < p_Recursion:
      return ["F"], 0
   elif choice < p_Recursion+p_Negation:
      f, v = constructFormula()
      return ["neg"] + f, (1-v)
   elif choice < 1-(1-p_Recursion+p_Negation)/2:
      f1, v1 = constructFormula()
      f2, v2 = constructFormula()
      return ["("] + f1 + ["AND"] + f2 + [")"], (v1*v2)
   elif choice < 1.0:
      f1, v1 = constructFormula()
      f2, v2 = constructFormula()
      return ["("] + f1 + ["OR"] + f2 + [")"], max(v1, v2)
   assert False

def constructFormulaPolish():
   choice = random()
   if choice < p_Recursion/2:
      return ["T"], 1
   elif choice < p_Recursion:
      return ["F"], 0
   elif choice < p_Recursion+p_Negation:
      f, v = constructFormula()
      return ["neg"] + f, (1-v)
   elif choice < 1-(1-p_Recursion+p_Negation)/2:
      f1, v1 = constructFormula()
      f2, v2 = constructFormula()
      return ["AND"] + f1 + f2, (v1*v2)
   elif choice < 1.0:
      f1, v1 = constructFormula()
      f2, v2 = constructFormula()
      return ["OR"] + f1 + f2, max(v1, v2)
   assert False

def constructFormulaPolishNand():
   choice = random()
   if choice < p_Recursion/2:
      return ["T"], 1
   elif choice < p_Recursion:
      return ["F"], 0
   elif choice < p_Recursion+p_Negation:
      f, v = constructFormula()
      return ["neg"] + f, (1-v)
   elif choice < 1.0:
      f1, v1 = constructFormula()
      f2, v2 = constructFormula()
      return ["NAND"] + f1 + f2, 1-(v1*v2)
   assert False



constructFormula = constructFormulaBrackets
#constructFormula = constructFormulaPolish
#constructFormula = constructFormulaPolishNand

data, label = [], []
while len(data) < 50000:
   f, v = constructFormula()
   if len(f) < 30 or len(f) > 40:
     continue
   data.append(f)
   #print(f)
   label.append(v)
   if len(data) % 500 == 0:
     print(len(f), f, v, len(data)/100000)
#quit()
#print(sum(label)/len(label))

averageLabel = sum(label)/len(label)

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
       self.output = torch.nn.Linear(100, 2)

       self.weights = torch.nn.Linear(1, numberOfFeatures)
       self.loss = torch.nn.CrossEntropyLoss(reduce=None, reduction="none")
       self.optimizer = optim.Adam(self.parameters(), lr = lr_lm)

       self.cnn = nn.Conv1d(400, 100, 5)

   def forward(self, current, printHere=False, computeAccuracy=False, train=True):
       global biasHead
       global crossEntropy
       targets = [x[1][1] for x in current]
       current = [x[1][0] for x in current]
       lengths = list(map(len, current))
       maxLength = max(lengths)
       input_words = []
       input_words = []
       #print(targets, current)
       input_words = [[encodeWord(x) for x in y] + [2 for _ in range(maxLength-len(y))] for y in current]

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       self.optimizer.zero_grad()

       input_words_padded = input_words
       words_layer = self.words_embeddings(Variable(torch.LongTensor(input_words_padded).to(device)))
       convolved = self.cnn(words_layer.transpose(1,2))
       convolved= self.relu(convolved)
       if train:
         convolved = self.dropout(convolved)
#       pooled = torch.max(convolved, dim=2)[0]
       pooled = torch.mean(convolved, dim=2)
       predicted = self.output(pooled)
       target = torch.LongTensor(targets).to(device)
       loss = self.loss(predicted, target)
       if printHere:
           print("Relation identification loss", float(loss.mean()))

       prediction = torch.where(predicted[:,0] < predicted[:,1], 1+0*target, 0*target)

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
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)


import torch.nn.functional

DEV_END = int(len(data)/10)
TEST_END = 2*int(len(data)/10)

data_test = data[DEV_END:TEST_END]
label_test = label[DEV_END:TEST_END]

data_dev = data[:DEV_END]
label_dev = label[:DEV_END]

data = data[TEST_END:]
label = label[TEST_END:]

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
  labeledData = list(zip(data, label))
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
           print("Running Average ", crossEntropy, accuracyMean, averageLabel, lastDevAcc)
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
           print(devLosses)
           print("Running Average ", crossEntropy, accuracyMean)
  except StopIteration:
    print("Next Epoch", counter*batchSize, epochs) 
  crossEntropyDev /= overallIndices
  accuracyMeanDev /= overallIndices

  if accuracyMeanDev > lastDevAcc:
     lastDevAcc = accuracyMeanDev
     devLosses.append(lastDevAcc)
  else:
     break


labeledData = list(zip(data_test, label_test))
shuffle(labeledData)
print("DATAPOINTS", len(labeledData))
labeledData = enumerate(labeledData)
crossEntropyTest = 0
accuracyMeanTest = 0
overallIndices = 0
try:
  while True:
     batch = [next(labeledData) for _ in range(batchSize)]
     counter += 1
     printHere = (counter % 100 == 0)
     loss, wordNum, accuracy = model.forward(batch, printHere, train=False)
#     print("TEST", float(loss)/batchSize, float(accuracy))
     decay = 0.9999 ** wordNum
     crossEntropyTest += float(loss)/wordNum
     accuracyMeanTest += float(accuracy)
     overallIndices += 1
     if printHere:
         print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
         print(devLosses)
         print("Running Average ", crossEntropy, accuracyMean)
except StopIteration:
  print("Next Epoch", counter*batchSize, epochs) 
crossEntropyTest /= overallIndices
accuracyMeanTest /= overallIndices

print("TEST",  "CE", crossEntropyTest,  "ACC", accuracyMeanTest, "Epochs", epochs)





 

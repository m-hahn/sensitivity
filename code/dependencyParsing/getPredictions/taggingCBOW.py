
import random
import sys

objectiveName = "graphParser"

print(sys.argv)



# Works well

import random
import sys
import re

objectiveName = "graphParser"

print(sys.argv)




import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="English-EWT_2.4")
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


from corpusIterator_V import CorpusIterator_V as CorpusIterator

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(args.language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum 

   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random
import numpy as np




def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]] for x in remainingChildren]
           if reverseSoftmax:
              logits = [-x for x in logits]
           softmax = logits #.view(1,-1).view(-1)
           selected = numpy.argmax(softmax)
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           

dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))



posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

vocab_deps.add("root")
itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))


itos_labels = itos_pos_uni
stoi_labels = stoi_pos_uni



print(itos_deps)

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)
for i, key in enumerate(itos_deps):
   dhLogits[key] = 2*(random()-0.5)
   dhWeights[i] = dhLogits[key]

   originalDistanceWeights[key] = random()  
   distanceWeights[i] = originalDistanceWeights[key]

import os




words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(zip(itos, range(len(itos))))


if len(itos) > 6:
   assert stoi[itos[5]] == 5



def toDevice(x):
   if True:
     return x
   else:
     return x.cuda()



from torch import optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 10.0
accuracyAverage = 0.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < args.vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)


import torch.nn.functional



baselinePerType = [4.0 for _ in itos_pure_deps]





from math import log, exp, sqrt
from random import random, shuffle, randint
import os



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


def encodeWord(w):
   assert w == w.lower()
   return stoi[w]+3 if w in stoi and stoi[w] < vocab_size else 1

from torch import optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import scipy.special

class Model(nn.Module):
   def __init__(self):
       super(Model, self).__init__()
       self.words_embeddings_center = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 300)
       self.words_embeddings_surround = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 300)
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
                 self.words_embeddings_center.weight.data[stoi[line[0]]+3] = torch.FloatTensor([float(x) for x in line[1:]])
                 self.words_embeddings_surround.weight.data[stoi[line[0]]+3] = torch.FloatTensor([float(x) for x in line[1:]])
 

       self.dropout = nn.Dropout(dropout_rate)
       self.inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)
       self.softmax_layer = torch.nn.Softmax()
       self.logsoftmax = torch.nn.LogSoftmax()
       self.logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)

       #self.first_layer = torch.nn.Linear(350, 400)
       self.relu = torch.nn.ReLU()
       self.output = torch.nn.Linear(256, len(itos_labels))
       self.hidden = torch.nn.Linear(300, 256)

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
#       print(current[0])
       targets = [stoi_labels[x[0]["posUni"]] for x in current]
       surrounds = [[y["word"] for y in x[1]] for x in current]
       centers = [encodeWord(x[0]["word"]) for x in current]
       lengths_surrounds = list(map(len, surrounds))
       maxLength_surrounds = max(lengths_surrounds)
       input_words_surrounds = [[encodeWord(x) for x in y] + [2 for _ in range(maxLength_surrounds-len(y))] for y in surrounds]
       self.optimizer.zero_grad()

       words_layer_surrounds = self.words_embeddings_surround(Variable(torch.LongTensor(input_words_surrounds).to(device)))
       words_layer_centers = self.words_embeddings_center(Variable(torch.LongTensor(centers).to(device)))
       #print(words_layer_surrounds.size(), words_layer_centers.size())
       words_layer_surrounds = torch.cat([words_layer_surrounds, words_layer_centers.unsqueeze(1)], dim=1)
       if train:
         words_layer_surrounds = self.dropout(words_layer_surrounds)
       representation_premise = words_layer_surrounds.mean(dim=1)
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






devLosses = []
devAccuracies = []

import sys


def computeDevLoss():
         global printHere
         counterDev = 0
         corpusDev = CorpusIterator(args.language, "dev", shuffleData=False).iterator(rejectShortSentences = True)
         partitionsDev = corpusIntoPartitions(corpusDev, includeSmallerBatch=True)
         devLoss = 0
         devWords = 0
         devAccuracy = 0
         for partitionDev in partitionsDev:
              counterDev += 1
              printHere = (counterDev % 500 == 0)
              loss, wordNum, accuracy = model.forward(partitionDev, train=False)
              devLoss += loss.data.cpu().numpy()
              devWords += wordNum
              devAccuracy += accuracy * wordNum
              if counterDev % 50 == 0:
                print("Run on dev "+str(counterDev))
                print((devLoss/devWords, devWords, devAccuracy/devWords))

         newDevLoss = devLoss/devWords        
         devLosses.append(newDevLoss)
         devAccuracies.append(devAccuracy/devWords)

WINDOW_AROUND_WORD = 5

def corpusIntoPartitions(corpus, includeSmallerBatch=False):
  corpus = list(corpus)
  corpus = sorted(corpus, key=len)
  #print(corpus[:10])
  partitions = []
  corpus = iter(corpus)
  partition = []
  while True:
    try:
      for _ in range(batchSize):
         sentence = next(corpus)
         for i in range(len(sentence)):
            partition.append((sentence[i], sentence[i-WINDOW_AROUND_WORD:i+WINDOW_AROUND_WORD]))
            if len(partition) == batchSize:
               partitions.append(partition)
               partition = []
    except StopIteration:
      if includeSmallerBatch and len(partition) > 0:
         partitions.append(partition)
      break
  shuffle(partitions)
  return partitions

counter = 0
epochs = 0
import math
crossEntropy = math.log(len(itos_deps))
while True:
  corpus = CorpusIterator(args.language, "train", shuffleData=False).iterator(rejectShortSentences = True)
  partitions = corpusIntoPartitions(corpus)
  epochs += 1
  for partition in partitions:
       counter += 1
       printHere = (counter % 100 == 0)
       loss, wordNum, accuracy = model.forward(partition, printHere)
       decay = 0.9999 ** wordNum
       crossEntropy = (1-decay) * float(loss)/wordNum + decay * crossEntropy
       accuracyAverage = (1-decay) * accuracy + decay * accuracyAverage 

       if wordNum == 0:
          assert loss is 0
       else:
          model.backward(loss)
       if printHere:
           print(args)
           print(" ".join(list(map(str,[__file__, args.language, myID, counter, epochs ]))))
           print(devLosses)
           print(devAccuracies)
           print("Running Average ", "CE", crossEntropy, "Accuracy", accuracyAverage)
       if counter % 500 == 0:
          print((myID, "EPOCHS", epochs, "UPDATES", counter, "perEpoch", devLosses, devAccuracies), file=sys.stderr)



  if True:
         print((myID, "EPOCHS", epochs, "UPDATES", counter), file=sys.stderr)

         computeDevLoss()



         if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
            del devLosses[-1]
            del devAccuracies[-1]
            print("Loss deteriorating, stop")
            print(args)

            with open("/u/scr/mhahn/sensitivity/parse-labeling/performance-"+args.language+"_"+__file__+"_"+str(myID)+".txt", "w") as outFile:
                 print(str(args), file=outFile)
                 print(" ".join(map(str,devLosses)), file=outFile)
                 print(" ".join(map(str,devAccuracies)), file=outFile)
            break

counts = {y : 0 for y in itos_labels}
sumCounts = 0
corpusDev = CorpusIterator(args.language, "dev", shuffleData=False).iterator(rejectShortSentences = True)
partitionsDev = corpusIntoPartitions(corpusDev, includeSmallerBatch=True)
for z in partitionsDev:
  for y in z:
    counts[y[0]["posUni"]] += 1
    sumCounts += 1

print("Majority class", max(y for _, y in counts.items())/sumCounts)



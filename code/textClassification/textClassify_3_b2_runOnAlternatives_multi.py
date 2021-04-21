# Works well

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
    label = train_y + test_y
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


NUMBER_OF_CLASSES = len(set(label))

word_embedding_size = 50
dropout_rate = 0.4
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
       self.output = torch.nn.Linear(100, NUMBER_OF_CLASSES)

       self.weights = torch.nn.Linear(1, numberOfFeatures)
       self.loss = torch.nn.CrossEntropyLoss(reduce=None, reduction="none")
       self.optimizer = optim.Adam(self.parameters(), lr = lr_lm)

       self.cnn = nn.Conv1d(400, 100, 1) 

   def forward(self, current, printHere=False, computeAccuracy=False, train=True, getPredictions=False):
       global biasHead
       global crossEntropy
#       if getPredictions:
 #         print("####")
  #        print(current[0])
       targets = [x[1][1] for x in current]
       current = [x[1][0] for x in current]
       lengths = list(map(len, current))
       maxLength = max(lengths)
       input_words = []
       input_words = []
       #print(targets, current)
   #    if getPredictions:
    #      print("====")
     #     print(current[0])
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

       prediction = torch.argmax(predicted, dim=1)

       if getPredictions:
         return torch.softmax(predicted, dim=1).detach() # return the probability vector
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
           print(devLosses)
           print("Running Average ", crossEntropy, accuracyMean)
  except StopIteration:
    print("Next Epoch", counter*batchSize, epochs) 
  crossEntropyDev /= overallIndices
  accuracyMeanDev /= overallIndices

  if accuracyMeanDev > lastDevAcc:
     lastDevAcc = accuracyMeanDev
  else:
     print("Reached early stopping")
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




with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}_alternatives.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

from transformers.data.processors.utils import InputExample

alternatives_examples = []

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   
   alternative = alternative.split("\n")
   print(alternative[:5])
   for line in alternative[2:]:
     if len(line) < 5:
       continue
     #print(line)
     if "\t" not in line:
        continue
     subset, sentence= line.strip().split("\t")
     sentence_unprocessed = sentence 
     result = [""]
     for word in sentence.strip().split(" "):
        if word.startswith("▁"):
            result.append(word[1:])
        else:
            result[-1] = result[-1] + word
     sentence = " ".join(result)
     sentence = sentence.replace("  ", " ")
     sentence = sentence.strip()
     if sentence not in variants_set:
     #  print(sentence)
       
       variants_set.add(sentence)
       sentence = dataloader.clean_str(sentence).split(" ")
       #print(sentence)
       alternatives_examples.append((sentence,sentence_unprocessed))


print(len(alternatives_examples))

alternatives_examples_ = enumerate(alternatives_examples)

predictions_total = torch.FloatTensor([])

print("line 383")

while True:
   try:
      batch = [[None, [next(alternatives_examples_)[1][0], 0]] for _ in range(batchSize)]
   except StopIteration:
      break
   counter += 1
   printHere = (counter % 100 == 0)
   predictions = model.forward(batch, printHere, train=False, getPredictions=True)
   predictions_total = torch.cat([predictions_total, predictions.cpu()], dim=0)
#   print(predictions)
predictionsBySentence = {}
print(predictions_total.float().mean())
for j in range(predictions_total.size()[0]):
 # print(alternatives_examples[j])
  predictionsBySentence[alternatives_examples[j][1].strip()] = (predictions_total[j])
#print(predictionsBySentence)
#quit()
print(alternatives_examples[:5])


#def mean(values):
 #  return sum(values)/len(values)

def variance(values):
   return values.pow(2).mean(dim=0) - values.mean(dim=0).pow(2)

from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun

from random import shuffle

#alternatives_predictions_binary = {}
#alternatives_predictions_float = {}
#with open(f"/u/scr/mhahn/PRETRAINED/textclas/{task}_alternatives_Predictions.txt", "r") as inFile:
#  for line in inFile:
#     if len(line) < 5:
#       continue
#     sentence, binary, cont = line.strip().split("\t")
#     alternatives_predictions_binary[sentence.strip()] = binary.strip()
#     alternatives_predictions_float[sentence.strip()] = cont.strip()
#  print(len(alternatives_predictions_binary))
#
#print(list(alternatives_predictions_binary.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}_alternatives.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
   #print(original)

   tokenized = alternative[1].split(" ")
   for variant in alternative[2:]:
      if len(variant) < 5:
         continue
      if "\t" not in variant:
         continue
      subset, sentence= variant.strip().split("\t")
       
      sentence = sentence.strip()

#      print(sentence+"#")
#      print(list(predictionsBySentence.items())[:10])
      if sentence not in predictionsBySentence:
#         print("NO PREDICTION", sentence)
         continue
      assert sentence in predictionsBySentence, sentence


      variants_set.add(sentence)
      if subset not in variants_dict:
         variants_dict[subset] = []
      variants_dict[subset].append(sentence)
  # print((result))
   print(len(variants_set), "variants")
   if len(variants_set) == 0:
      continue
   valuesPerVariant = {}
   for variant in variants_set:
   #  print(variant)
#     try:
#       assert alternatives_predictions_binary[variant] in ["0", "1"], alternatives_predictions_binary[variant]
#       valuesPerVariant[variant] = 1 if alternatives_predictions_binary[variant] == "1" else -1
     #   print(variant)
        valuesPerVariant[variant] = predictionsBySentence[variant]
     #  if len(valuesPerVariant) % 100 == 0:
      #   print(valuesPerVariant[variant], valuesPerVariant[variant] == True, len(valuesPerVariant), len(variants_set), variant)
 #    except ValueError:
  #      print("VALUE ERROR", variant)
   #     valuesPerVariant[variant] = 0
    # except AttributeError:
     #   print("VALUE ERROR", variant)
      #  valuesPerVariant[variant] = 0

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.stack([ valuesPerVariant[x] for x in variants_dict[subset]], dim=0)
       varianceBySubset[subset] = variance(values).sum() # variance summed over the elements of the probability vector


   subsetsEnumeration = list(variants_dict)
    
   N = len(subsetsEnumeration[0])
   A = [[0 for subset in range(len(subsetsEnumeration))] for inp in range(N)]
   for inp in range(N):
       for subset, bitstr in enumerate(subsetsEnumeration):
          assert len(bitstr) == N
          if bitstr[inp] == "1":
              A[inp][subset] = 1
   
   
   b = [1 for _ in range(N)]
   x_bounds = [(0,1) for _ in range(len(subsetsEnumeration))]
   perSubsetSensitivities = [varianceBySubset[x] for x in subsetsEnumeration]

   sensitivity = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))


import sys
import torch

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def oneHotVector(i):
   vec = torch.zeros(len(stoi_upos))
   vec[stoi_upos[i]] = 1
   return vec


#def variance(values):
#   if True: # continuous
#       values = torch.FloatTensor([[float(x) for x in y.split(" ")] for y in values])
##       print(values.sum(dim=1))
# #      quit()
#       values = 2*values-1
#       variancePerType =  ((values - values.mean(dim=0)).pow(2)).mean(dim=0)
#       #print(variancePerType)
#       #quit()
#       return float(variancePerType.max())
#
#   itos = list(set(values))
#   if len(itos) == 1:
#      return 0
#   stoi = dict(zip(itos, range(len(itos))))
#   oneHotVectors = torch.zeros(len(values), len(itos)) - 1
#   for i in range(len(values)):
#       oneHotVectors[i][stoi[values[i]]] = 1
#   #print(oneHotVectors)
#   variancePerType =  ((oneHotVectors - oneHotVectors.mean(dim=0)).pow(2)).mean(dim=0)
#   #print(variancePerType)
#   #quit()
#   return float(variancePerType.sum())

from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun

from random import shuffle

upos_set = set()

sentenceToWord = {}
alternatives_predictions_categorical = {}
alternatives_predictions_continuous = {}
with open(f"/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_PMLM_1billion_raw_predictions.txt", "r") as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     sentence, word, head, relation, upos, relation_vector = line
     if len(word) == 0:
      continue
 #    print(line)
#     if word[0] != "▁":
 #      continue
     alternatives_predictions_categorical[sentence.strip()] = upos
     alternatives_predictions_continuous[sentence.strip()] = None
     upos_set.add(upos)
     sentenceToWord[sentence.strip()] = word
  print(len(alternatives_predictions_categorical))

stoi_upos = dict(list(zip(list(upos_set), range(len(upos_set)))))


from collections import defaultdict
alternatives_PMLM = defaultdict(list)
with open("/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_PMLM_1billion_raw.tsv", "r") as inFile:
  for line in inFile:
     mask, original, alternative = line.strip().split("\t")
#     print((mask, original, alternative))
     mask = mask.strip()
     original = original.strip()
     alternative = alternative.strip()
#     if mask == "00001111111100000000000000":
 #      quit()
     alternatives_PMLM[(original.strip(), mask.strip())].append(alternative.strip())
     if mask == "00000100":
       print((original, mask))
assert ('▁From ▁the ▁AP ▁comes ▁this ▁story ▁ :', '00000100') in alternatives_PMLM
#quit()

print(list(alternatives_predictions_categorical.items())[:10])

with open(f"/juice/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_c.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

with open(f"/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinaryS1ensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
   print(original)

   tokenized = alternative[1].strip()
   original = ("".join(tokenized)).replace("▁", " ").strip()
#   if original not in alternatives_predictions_continuous:
#      print("MISSING ORIGINAL", original)
#      continue
#   originalPrediction = torch.FloatTensor([float(y) for y in alternatives_predictions_continuous[original].split(" ")])
   hasEvaluatedSubset = set()
   for variant in alternative[2:]:
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      subset = subset.strip()
      if subset in hasEvaluatedSubset:
         continue
      hasEvaluatedSubset.add(subset)
      
#      if sentence not in alternatives_predictions_categorical:
#         assert False, sentence
 #        continue
      #assert sentence in alternatives_predictions_categorical, sentence
      relevantVariants = alternatives_PMLM[(tokenized.strip(), subset.strip())]
      if len(relevantVariants) == 0:
        print("ODD", (tokenized.strip(), subset.strip()))
        continue
#        assert False
      for var in relevantVariants:
         if subset not in variants_dict:
            variants_dict[subset] = []
         var = var.replace("[CLS]", "").split("[SEP]")[0].strip()
         if var not in alternatives_predictions_continuous:
            print("ODD", var)
            continue
         variants_dict[subset].append(var)
#         print(sentenceToWord[var]) # this is for sanity-checking
   variants_set = set(list(variants_dict))
   print(len(variants_set), "variants")
   if len(variants_set) == 0:
     continue
   varianceBySubset = {}
   for subset in variants_dict:
       if len(variants_dict[subset]) == 0:
           varianceBySubset[subset] = 0
           continue
            
       values = torch.stack([oneHotVector(alternatives_predictions_categorical[x]) for x in variants_dict[subset]], dim=0)
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       if str(varianceBySubset[subset]) == "nan":
          varianceBySubset[subset] = 0
#       if varianceBySubset[subset] > 2:
 #         print(variants_dict[subset])
  #        print(values)
      
#   print(varianceBySubset)


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
   perSubsetSensitivities = [varianceBySubset.get(x, 0) + (0.00001 * sum([1 for q in x if q == "1"])) for x in subsetsEnumeration]

   sensitivity = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   sensitivityHistogram[int(2*sensitivity)] += 1
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
   print(original, "\t", sensitivity, file=outFile)

print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])


sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())



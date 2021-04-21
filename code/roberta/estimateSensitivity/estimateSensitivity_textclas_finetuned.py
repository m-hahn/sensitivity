
import sys
task = sys.argv[1]

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def variance(values):
   return mean([x**2 for x in values]) - mean(values)**2

from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun

from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}
with open(f"/u/scr/mhahn/PRETRAINED/textclas/{task}_alternatives_finetuned_Predictions.txt", "r") as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence, binary, cont = line
     alternatives_predictions_binary[sentence.strip()] = binary.strip()
     alternatives_predictions_float[sentence.strip()] = cont.strip()
  print(len(alternatives_predictions_binary))

print(list(alternatives_predictions_binary.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/textclas/{task}_alternatives_finetuned.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))
try:
  with open(f"/u/scr/mhahn/PRETRAINED/textclas/{task}_alternatives_finetuned_d.txt", "r") as inFile:
    alternatives += inFile.read().strip().split("#####\n")
    print(len(alternatives))
except FileNotFoundError:
  pass

sensitivities = []

with open(f"/u/scr/mhahn/sensitivity/sensitivities/sensitivities_{__file__}_{task}", "w") as outFile:
 print("Original", "\t", "BinarySensitivity", "\t", "FloatSensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
   print(original)

   tokenized = alternative[1].split(" ")
   for variant in alternative[2:]:
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
       
      result = [""]
      for word in sentence.strip().split(" "):
         if word.startswith("▁"):
             result.append(word[1:])
         else:
             result[-1] = result[-1] + word
      sentence = " ".join(result)
      sentence = sentence.replace("  ", " ")
      sentence = sentence.strip()

      if sentence not in alternatives_predictions_binary:
         continue
      assert sentence in alternatives_predictions_binary, sentence


      variants_set.add(sentence)
      if subset not in variants_dict:
         variants_dict[subset] = []
      variants_dict[subset].append(sentence)
  # print((result))
   print(len(variants_set), "variants")
   valuesPerVariant_binary = {}
   valuesPerVariant_float = {}
   for variant in variants_set:
   #  print(variant)
     try:
       assert alternatives_predictions_binary[variant] in ["0", "1"], alternatives_predictions_binary[variant]
       valuesPerVariant_binary[variant] = 1 if alternatives_predictions_binary[variant] == "1" else -1
       valuesPerVariant_float[variant] = float(alternatives_predictions_float[variant] )
     #  if len(valuesPerVariant) % 100 == 0:
      #   print(valuesPerVariant[variant], valuesPerVariant[variant] == True, len(valuesPerVariant), len(variants_set), variant)
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant_binary[variant] = 0
        valuesPerVariant_float[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant_binary[variant] = 0
        valuesPerVariant_float[variant] = 0
   varianceBySubset_binary = {}
   varianceBySubset_float = {}
   for subset in variants_dict:
       values_binary = [ valuesPerVariant_binary[x] for x in variants_dict[subset]]
       values_float = [ valuesPerVariant_float[x] for x in variants_dict[subset]]
       #print(subset, mean(values), variance(values))
       varianceBySubset_binary[subset] = variance(values_binary)
       varianceBySubset_float[subset] = variance(values_float)
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
   perSubsetSensitivities_binary = [varianceBySubset_binary[x] for x in subsetsEnumeration]
   sensitivity_binary = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities_binary)

   perSubsetSensitivities_float = [varianceBySubset_float[x] for x in subsetsEnumeration]
   sensitivity_float = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities_float)


   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity_binary, sensitivity_float)
   sensitivityHistogram[int(2*sensitivity_binary)] += 1
   sensitivities.append(sensitivity_binary)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
   print(original, "\t", sensitivity_binary, "\t", sensitivity_float, file=outFile)

print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])

import torch
sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())



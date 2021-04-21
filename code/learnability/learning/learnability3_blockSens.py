# Using ../sensitivity/36test.py

# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

# Uses BlockS1ensitivity


import torch
import random
from math import log

print("done loading")
N = 7

inputs = range(2**N)

TEMPERATURE = 1
weights = 0*torch.rand(2**N)**TEMPERATURE+1e-10 #**0 #**10 #** 2
weights = weights / weights.sum()

#print(weights)


# https://wiki.python.org/moin/BitManipulation
def parityOf(int_type):
        parity = 0
        while (int_type):
            parity = ~parity
            int_type = int_type & (int_type - 1)
        return(parity)

bitParity = []
for i in range(2**N):
   bitParity.append(parityOf(i)*2+1)
#print(bitParity)



A = [[0 for subset in range(2**N)] for inp in range(N)]
for inp in range(N):
    for subset in range(2**N):
       bitstr = format(subset, "b")
       bitstr = ("0"*(N-len(bitstr))) + bitstr
       assert len(bitstr) == N
       if bitstr[inp] == "1":
           A[inp][subset] = 1

b = [1 for _ in range(N)]
x_bounds = [(0,1) for _ in range(2**N)]
from scipy.optimize import linprog


def getMaxOverPartitions(perSubsetSensitivities, sensitivityObtainedSoFar, partitionSoFar, lastAdded):
#   print("{0:b}".format(2**N-1), "{0:b}".format(partitionSoFar), "{0:b}".format(2**N-1 & (~ partitionSoFar)), 2**N-1 & (~ partitionSoFar), lastAdded)
#   print("{0:b}".format(partitionSoFar), "{0:b}".format(2**N-1), partitionSoFar == 2**N-1)
   if partitionSoFar == 2**N-1:
      return sensitivityObtainedSoFar
   if (2**N-1) & (~ partitionSoFar) < lastAdded:
      return 0

   sens = 0
   for j in range(lastAdded+1, 2**N):
      # check whether j is in the complement of partitionSoFar
#      if perSubsetSensitivities[j] == 0:
 #         continue
#      print("{0:b}".format(j), "{0:b}".format(partitionSoFar), "{0:b}".format((~j | ~partitionSoFar)))

      if ~j | ~partitionSoFar == -1:

          here = getMaxOverPartitions(perSubsetSensitivities, sensitivityObtainedSoFar+perSubsetSensitivities[j], partitionSoFar | j, j)
          if here > sens:
              sens=here
   return sens




def getFractionalMaxOverPartitions(perSubsetSensitivities, sensitivityObtainedSoFar, partitionSoFar, lastAdded):
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   return -res.fun


# we can first express the standard basis in terms of bit strings
standardBasis = torch.Tensor([[bitParity[i&j] for i in range(2**N)] for j in range(2**N)])
#print(standardBasis.size())

def product(x,y):
    return (weights * x * y).sum()

def degree(subset):
   return len([y for y in "{0:b}".format(subset) if y == "1"])


subsetsPerDegree = [[] for _ in range(N+1)]
for i in range(2**N):
   subsetsPerDegree[degree(i)].append(i)

numberingOfBasis = sorted(range(2**N), key=lambda x:len([y for y in "{0:b}".format(x) if y == "1"]))
numberingOfBasisInverted = dict(list(zip(numberingOfBasis, range(len(numberingOfBasis)))))
#print(numberingOfBasis)

with torch.no_grad():
   orthonormalBasis = []
   correspondingSet = []
   
   for setHere, vector in [(i,standardBasis[i]) for i in numberingOfBasis]:
      vec = vector
      # Assuming uniform distribution
#      for u in orthonormalBasis:
 #        print(product(vector, u))
  #       vec = vec - product(vector, u) * u
      orthonormalBasis.append(vec) # / product(vec,vec).sqrt()) 
      correspondingSet.append(setHere)
#   print(orthonormalBasis)
#print(product(orthonormalBasis[2], orthonormalBasis[2]))
#print(product(standardBasis[2], standardBasis[2]))

basisElementsPerDegree = [[] for _ in range(N+1)]
for index, vec in enumerate(orthonormalBasis):
    degree = len([y for y in "{0:b}".format(correspondingSet[index]) if y == "1"])
    basisElementsPerDegree[degree].append((index, vec))


import math
# randomize the function using the orthonormal basis

def sampleFunction(fourierMean):
#    function = (torch.randint(low=0, high=2, size=(2**N,))*2-1).float()

    function = 0*(torch.randint(low=0, high=2, size=(2**N,))*2-1).float() 
    contributionPerDegree = torch.FloatTensor([random.random() for _ in range(N+1)]) # these contributions will be actual squared L2 weights for the subspaces
    contributionPerDegree[0] = 0

    for i in range(1, N+1):
       if abs(i-fourierMean) > 1:
           contributionPerDegree[i] = 0

    contributionPerDegree = contributionPerDegree / contributionPerDegree.sum()
    for degree in range(len(contributionPerDegree)):
        contributionPerElements = torch.FloatTensor([random.random()-0.5 for _ in basisElementsPerDegree[degree]])
        contributionPerElements.div_(contributionPerElements.pow(2).sum().sqrt()).mul_(contributionPerDegree[degree].sqrt())
    #    print(contributionPerElements.pow(2).sum(), contributionPerDegree[degree])
        for i in range(len(contributionPerElements)):
            function += contributionPerElements[i] * basisElementsPerDegree[degree][i][1]
     #   print(degree, getWeightsByDegree(function/function.pow(2).mean().sqrt()))
    function = function / function.pow(2).mean().sqrt()
#    print("NORM", function.pow(2).mean())

#    print("Overall variance", (function - function.mean()).pow(2).mean())
#    averageDegree, _ = getWeightsByDegree(function)
    averageDegree = (contributionPerDegree * torch.FloatTensor(range(N+1))).sum()

    #print("CONTRIBUTION PER DEGREE", contributionPerDegree, averageDegree)

    return function, averageDegree



def computeAverageBlockSensitivity(function):
    totalAverageBlockSensitivity = 0
    for input_string in range(2**N):
     #print(input_string)
     perSubsetSensitivities = [0 for _ in range(2**N)]
     #print("==================")
     #print("       input string", format(input_string, "b"))
     for subset in range(1,2**N):
      #  print("subset", format(subset, "b"))
        totalProbability = 0
        sumValue = 0
        sumSquaredValue = 0
        # now take all possible combinations for the bits in subset   
        for i in range(subset+1):
          # check whether i is a subset of subset
          if ((~i) | subset) == -1:
    #         print("    candidate", format(i, "b"))
      #       print("{0:b}".format(subset), "{0:b}".format(i), "{0:b}".format(((~i) | subset)))
             candidate = input_string ^ i
             value = float(function[candidate])
             probHere = float(weights[candidate])
   #          print("       changed input", format(candidate, "b"), "function value", value, "prob", probHere)
             sumValue += value * probHere
             sumSquaredValue += (value ** 2) * probHere
             totalProbability += probHere
        if totalProbability == 0: 
           variance = 0
        else:
           variance = sumSquaredValue/totalProbability - (sumValue/totalProbability)**2
    #    print(variance)
    #    print(input_string, subset, probs, prob,entropy)
        perSubsetSensitivities[subset] = float(variance)
  #      print("SENSITIVITY HERE", float(weights[input_string]) * variance)
 #   print(perSubsetSensitivities)
    #quit() 
    # now find the partition that yields the greatest summed sensitivity
     partition = 0
#     print("per subset sensitivities", perSubsetSensitivities)
     bestFractional = getFractionalMaxOverPartitions(perSubsetSensitivities, 0, 0, 0)
     #bestHard = getMaxOverPartitions(perSubsetSensitivities, 0, 0, 0)
     #assert bestHard <= bestFractional+1e-5, (bestHard, bestFractional)
     #print(bestHard/bestFractional)
     best = bestFractional
    #  print(best)
     totalAverageBlockSensitivity +=  weights[input_string] * best
    return totalAverageBlockSensitivity






def getWeightsByDegree(function):
    weightByDegree = [0 for _ in range(N+1)]
    coefficients = []
    for index, vec in enumerate(orthonormalBasis):
        coefficient = float(product(vec, function))
        degree = len([y for y in "{0:b}".format(correspondingSet[index]) if y == "1"])
   #     print("{0:b}".format(correspondingSet[index]), degree,"\t", float(coefficient**2))
        coefficients.append(coefficient)
        weightByDegree[degree] += float(coefficient**2)
#    print(sum([x**2 for x in coefficients]))
    print("WEIGHT BY DEGREE", weightByDegree)
    averageDegree = sum([x*y for x,y in zip(weightByDegree, range(N+1))])
    return averageDegree, weightByDegree




import torch
from torchqrnn import QRNN
import random

seq_len, batch_size, hidden_size = N, 32, 128 # 128 used to be 32 (figure in the current draft)
size = (seq_len, batch_size, hidden_size)


def makeBitTensor(x):
  y = format(x, "b")
  y = ("0"*(N-len(y))) + y
  return [int(z) for z in list(y)]



def fitNetwork(function):
   embeddings = torch.nn.Embedding(2, hidden_size).cuda()
   
   qrnn = torch.nn.LSTM(hidden_size, hidden_size, num_layers=1).cuda()
   
   output = torch.nn.Linear(hidden_size, 1, bias=False).cuda()
   
   tanh = torch.nn.Tanh()
   
   def parameters():
     for x in [embeddings, qrnn, output]:
       for y in x.parameters():
          yield y
   
   optimizer = torch.optim.Adam(parameters(), lr=0.0003)
   
   movAvg = 0
   lossesAfterIterations = [] 
   for iteration in range(100000):
     optimizer.zero_grad()
     inputs = [random.randint(0, 2**N-1) for _ in range(batch_size)]
     targets = torch.FloatTensor([float(function[x]) for x in inputs]).cuda()
     inputNum = torch.LongTensor([makeBitTensor(x) for x in inputs]).cuda().t()
     _, hidden = qrnn(embeddings(inputNum))
     result = (output(hidden[0][-1])).view(-1)
#     print(result.size())
     result = result.view(batch_size)
     loss = (result - targets).pow(2).mean()
     movAvg = 0.95 * movAvg + (1-0.95) * (float(loss))
     if iteration % 10 == 0 and False:
       print(movAvg / (1-0.95**(iteration+1)), seq_len)
     (loss).backward()
     optimizer.step()
#     print(iteration, abs(log(iteration+1)/log(10) % 1))
     if iteration in [99, 999, 9999, 99999, 999999] : #(iteration+1) %  100 == 0:
        print(iteration)
        lossesAfterIterations.append(movAvg)
   return lossesAfterIterations 

with open(f"output/losses_{__file__}.tsv", "w") as outFile:
  print("\t".join(["AverageDegree", "AverageBlockSensitivity", "Acc100", "Acc1000", "Acc10000", "Acc100000"]), file=outFile)
  for _ in range(5):
   for j in range(0, N+2):
    function, averageDegree = sampleFunction(j) 
    averageBlockSensitivity = float(computeAverageBlockSensitivity(function))
    print(averageDegree, averageBlockSensitivity)
    loss = fitNetwork(function) 
    print(loss, averageDegree, loss)
    print("\t".join([str(x) for x in ([float(averageDegree), averageBlockSensitivity] + loss)]), file=outFile)

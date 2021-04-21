with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
  roberta = [x.split("\t") for x in inFile.read().strip().split("\n")]
  roberta = dict([(x[0].strip(), x) for x in roberta])
   

with open("/u/scr/mhahn/sensitivity/sensitivities/s31ensitivities_estimateS31ensitivity_SST2.py", "r") as inFile:
  s31ensitivities = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header_s31ensitivities = s31ensitivities[0]
  s31ensitivities = dict([(x[0].strip(), x) for x in s31ensitivities[1:]])

with open("/u/scr/mhahn/sensitivity/sensitivities/sensitivities_estimateSensitivity_SST2.py", "r") as inFile:
  sensitivities = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header_sensitivities = sensitivities[0]
  sensitivities = dict([(x[0].strip(), x) for x in sensitivities[1:]])

with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev-labels-1CNN.tsv", "r") as inFile:
   bowPerformance =  [x.split("\t") for x in inFile.read().strip().split("\n")]
   header_bowPeformance = bowPerformance[0]
   bowPerformance = dict([(x[0].strip(), x) for x in bowPerformance[1:]])

def getText(tree):
#   print(tree.children, tree.text)
   if tree.text is not None:
      return tree.text
   text = " ".join([getText(y) for y in tree.children])
   return text

features = {}
import pytreebank
dataset = pytreebank.load_sst()
for example in dataset["dev"]:
#   print(example.text in sensitivities, example.text)
   text = getText(example).lower()
 #  print("#"+text+"#")
   if (text in sensitivities):
      print(text, len(features), len(sensitivities))
      subspansBySentiment = {x:0 for x in ["negative", "neutral", "positive"]}
      for label, sentence in example.to_labeled_lines():
        subspansBySentiment[["negative", "negative", "neutral", "positive", "positive"][label]] += 1
      features[text] = {"subspans_"+x : subspansBySentiment[x] for x in subspansBySentiment}


with open(f"../outputs/{__file__}.tsv", "w") as outFile:
 print("\t".join(["Sentence", "Length", "Label", "BOWPrediction", "FloatSensitivity", "FloatS31ensitivity", "SubspansPos", "SubspansNeut", "SubspansNeg", "RobertaPrediction"]), file=outFile)
 for x in sensitivities:
  if x not in bowPerformance:
    print("MISSING", x)
    continue
#  print(bowPerformance[x], sensitivities[x])
  print("\t".join([str(y) for y in [x, len(x.split(" ")), bowPerformance[x][1], bowPerformance[x][2], sensitivities[x][1], s31ensitivities[x][1], features[x]["subspans_positive"], features[x]["subspans_neutral"], features[x]["subspans_negative"], roberta[x][3]]]), file=outFile)
print(len(sensitivities))

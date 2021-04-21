with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
  roberta = [x.split("\t") for x in inFile.read().strip().split("\n")]
  roberta = dict([(x[0].strip(), x) for x in roberta])
   

with open("/u/scr/mhahn/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_SST2.py", "r") as inFile:
  s3ensitivities = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header_s3ensitivities = s3ensitivities[0]
  s3ensitivities = dict([(x[0].strip(), x) for x in s3ensitivities[1:]])

with open("/u/scr/mhahn/sensitivity/sensitivities/sensitivities_estimateSensitivity_SST2.py", "r") as inFile:
  sensitivities = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header_sensitivities = sensitivities[0]
  sensitivities = dict([(x[0].strip(), x) for x in sensitivities[1:]])

with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev-labels-LSTM.tsv", "r") as inFile:
   lstmPerformance =  [x.split("\t") for x in inFile.read().strip().split("\n")]
   header_lstmPeformance = lstmPerformance[0]
   lstmPerformance = dict([(x[0].strip(), x) for x in lstmPerformance[1:]])

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
 print("\t".join(["Sentence", "Label", "LSTMPrediction"]), file=outFile)
 for x in sensitivities:
  if x not in lstmPerformance:
    print("MISSING", x)
    continue
#  print(lstmPerformance[x], sensitivities[x])
  print("\t".join([str(y) for y in [x, lstmPerformance[x][1], lstmPerformance[x][2]]]), file=outFile)
print(len(sensitivities))

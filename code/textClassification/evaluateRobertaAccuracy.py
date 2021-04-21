with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/CoLA/processed/dev.label", "r") as inFile:
   ground = inFile.read().strip().split("\n")

with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/CoLA/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
   predicted = [x.split("\t")[-1] for x in inFile.read().strip().split("\n")]

print(sum([1 if x==y else 0 for x, y in zip(ground, predicted)])/len(ground))


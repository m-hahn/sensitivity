with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_alternatives.tsv", "r") as inFile:
    alternatives = inFile.read().strip().split("#####\n")
processed = set()

with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_datapoints_raw.tsv", "w") as outFile:
  for alt in alternatives:
    alt = alt.strip().split("\n")
    if len(alt) < 3:
     continue
    sent = alt[2]
    if True:
       sent = sent.replace("</s>", "").replace(" ", "").replace("▁", " ").strip()
       if sent not in processed:
          processed.add(sent)
          print(sent, file=outFile)
          if sent.endswith(" himself."):
              print(sent.replace(" himself.", " themselves."), file=outFile)
          elif sent.endswith(" themselves."):
              print(sent.replace(" themselves.", " himself."), file=outFile)
          else:
              assert False
            

from nltk.tokenize.treebank import TreebankWordDetokenizer                                         
detokenizer = TreebankWordDetokenizer()                      

with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_248_alternatives_PMLM_1billion_raw.tsv", "r") as inFile:
    alternatives = inFile.read().strip().split("\n")
processed = set()

with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_248_alternatives_PMLM_1billion_raw_forGPT2.tsv", "w") as outFile:
  for alt in alternatives:
    alt = alt.strip().split("\t")
    _, _, sent = alt
    if sent not in processed:
          processed.add(sent)
          sent = detokenizer.detokenize( sent.strip().replace("[CLS]", "").split("[SEP]")[0].strip().split(" "))
          print(sent, file=outFile)
          if sent.endswith(" himself."):
              print(sent.replace(" himself.", " themselves."), file=outFile)
          elif sent.endswith(" themselves."):
              print(sent.replace(" themselves.", " himself."), file=outFile)
          else:
              print(sent, alt)
              continue
            

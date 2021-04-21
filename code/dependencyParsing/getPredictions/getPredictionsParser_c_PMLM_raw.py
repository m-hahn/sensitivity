from nltk.tokenize.treebank import TreebankWordDetokenizer         
detokenizer = TreebankWordDetokenizer()                            
                                                                   

#/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives.txt
A = False
sentences = set()
relevantWordSoFar = None
with open("/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_PMLM_1billion_raw.tsv", "r") as inFile:
   for line in inFile:
     line = line.strip()
     _, tokenized, modified = line.strip().split("\t")
     tokenized = tokenized.strip().split(" ")
     relevantWord_ = tokenized[int(len(tokenized)/2)]
     sentence = tokenized
     relevantWordSoFar = None
      
     relevantWord_ = (sentence[int(len(sentence)/2)])
     relevantWord = (sentence[int(len(sentence)/2)]).replace("▁", "")
     modified = modified.replace("[CLS]", "").replace("[SEP]", "").strip().split(" ")
     positions = [i for i in range(len(modified)) if (modified[i].startswith(relevantWord) if relevantWord_.startswith("▁") else relevantWord in modified[i])]
     positions = sorted(positions, key=lambda x:abs(x-len(modified)/2), reverse=True)
     if len(positions) == 0:
       print("ERROR", modified, relevantWord)
       continue
     sentenceMarked = (" ".join(modified[:positions[0]])) + "###" + (" ".join(modified[positions[0]:]))
     sentence = " ".join(modified)
     sentences.add((sentence, relevantWord, sentenceMarked.index("###")))
  #   print(sentenceMarked, relevantWord)
 #    print(sentence, relevantWord)
#     quit()
     if "Balkans" in line:
         print(sentence)



#assert "From the Balkans comes this story :" in sentences
#quit()
sentences = list(sentences)
print(sentences[:10])
print(len(sentences))

print("NOTE: Somehow, about 74 sentences end up not appearing in the predictions file. An example is: From the Balkans comes this story :")


sentences_ = sentences

import stanza
stanza.download('en')       # This downloads the English models for the neural pipeline
nlp = stanza.Pipeline('en', processors="tokenize,pos,lemma,depparse") # This sets up a default neural pipeline in English


with open("/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_PMLM_1billion_raw_predictions.txt", "w") as outFile:
 for q in range(int(len(sentences_)/100)):
   sentences = sentences_[q*100:(q+1)*100]
   #sentences = sentences[:100]
   
   def parseCharOffsets(misc):
      return [int(x.split("=")[1]) for x in misc.split("|")]
   
   joinedSentences = "\n\n".join([x[0] for x in sentences])
   doc = nlp(joinedSentences)
   print(len(doc.sentences), len(sentences))
   i_doc = 0 # counting documents and tokens
   print(sentences[0])
   sentenceBoundaries = [0]
   for i in range(len(joinedSentences)):
     if joinedSentences[i:i+2] == "\n\n":
        sentenceBoundaries.append(i)
   sentenceBoundaries.append(len(joinedSentences))
   oldSentenceThisBelongsTo = -1
   sentencesParsed = [[] for _ in sentences]
   for sentence in doc.sentences:
     for word in sentence.words:
        text = word.text
        char_pos = parseCharOffsets(word.misc)
        sentenceThisBelongsTo = sentenceBoundaries.index([x for x in sentenceBoundaries if x >= char_pos[0]][0])-1
        #print(sentenceThisBelongsTo)
        if sentenceThisBelongsTo != oldSentenceThisBelongsTo:
  #         print(sentences[sentenceThisBelongsTo])
           #print(sentenceThisBelongsTo)
           oldSentenceThisBelongsTo = sentenceThisBelongsTo
           assert oldSentenceThisBelongsTo == sentenceThisBelongsTo
        #print(text, joinedSentences[char_pos[0]:char_pos[1]], sentenceThisBelongsTo)
   #     assert sentenceThisBelongsTo >= 0
        sentencesParsed[max(0,sentenceThisBelongsTo)].append((word))
   #quit()
   
#   print(i_doc)
   for i in range(len(sentences)):
      #  print(sentencesParsed[i])
     #   print(sentences[i])
        characterLengthAccumulated = 0
        startChar = parseCharOffsets(sentencesParsed[i][0].misc)[0]
        FOUND = False
        #print(sentencesParsed[i])
        #print(sentences[i])
        for j in range(len(sentencesParsed[i])):
          characterLengthAccumulated = parseCharOffsets(sentencesParsed[i][j].misc)[0] - startChar
        #  print(i, sentencesParsed[i][j].text, characterLengthAccumulated, sentences[i][1], sentences[i][2])
          if not FOUND and characterLengthAccumulated >= sentences[i][2]:
             if not sentencesParsed[i][j].text.startswith(sentences[i][1].replace("▁", "")):
                 print("NOTE", q, i, sentencesParsed[i][j].text, characterLengthAccumulated, sentences[i][1], sentences[i][2])
             if sentences[i][1].startswith("▁"):
                assert sentencesParsed[i][j].text.startswith(sentences[i][1].replace("▁", ""))
             head = sentencesParsed[i][j].head
             upos = sentencesParsed[i][j].upos
             deprel = sentencesParsed[i][j].deprel
             deprel_probs = sentencesParsed[i][j].deprel_probs
             assert head > 0 or deprel == "root"
             print("\t".join([sentences[i][0], sentences[i][1], str(head-j) if head > 0 else "ROOT", deprel, upos, " ".join([str(float(x)) for x in deprel_probs])]), file=outFile)
             FOUND = True
             break    
        if not FOUND:
              print("ERROR", sentencesParsed[i])
     
     
     
     
     
      

#/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives.txt
A = False
sentences = set()
relevantWordSoFar = None
with open("/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives.txt", "r") as inFile:
 try:
   while True:
     line = next(inFile).strip()
     if line == "#####":
        original = (next(inFile))
        tokenized = (next(inFile))
        tokenized = tokenized.strip().split(" ")
        relevantWord_ = tokenized[int(len(tokenized)/2)]
#        if A:
 #         break
  #      A = True
        line = next(inFile).strip()
        relevantWordSoFar = None
     sentence = line.strip().split("\t")[1].strip().split(" ")
     relevantWord = (sentence[int(len(sentence)/2)])
     if relevantWordSoFar is not None:
        assert relevantWord == relevantWordSoFar, (relevantWord, relevantWordSoFar, sentence, original, tokenized)
     else:
        relevantWordSoFar = relevantWord
     assert relevantWord == relevantWord_, (relevantWord, relevantWord_)
     sentenceMarked = "".join(sentence[:int(len(sentence)/2)]) + "###" + "".join(sentence[int(len(sentence)/2):])
     sentenceMarked = sentenceMarked.replace("▁", " ").strip()
     sentence = ("".join(sentence)).replace("▁", " ").strip()
     sentences.add((sentence, relevantWord, sentenceMarked.index("###")))
     if "Balkans" in line:
         print(sentence)

 except StopIteration:
   pass
#quit()


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


with open("/u/scr/mhahn/PRETRAINED/parsing/en_ewt-ud-dev_alternatives_predictions.txt", "w") as outFile:
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
     
     
     
     
     
      

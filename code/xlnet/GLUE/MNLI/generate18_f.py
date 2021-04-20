#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

originalsProcessed = set()
for part in ["c", "d", "e"]:
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MNLI/dev_alternatives_{part}.tsv", "r") as inFile:
   data = inFile.read().split("#####\n")[:-1]
   for x in data:
      if len(x) == 0:
        continue
      print(len(x))
      originalsProcessed.add(x[:x.index("\n")])
print(originalsProcessed)
#quit()

# Based on https://raw.githubusercontent.com/huggingface/transformers/master/examples/run_generation.py

# Construct neighbors for MNLI

# ~/python-py37-mhahn generate18_f.py --model_type="XLNet" --model_name_or_path=xlnet-large-cased --dataset=MNLI

import argparse
import logging

import numpy as np
import torch

from transformers import (
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos> """


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text



PREPROCESSING_FUNCTIONS = {
    "xlnet": prepare_xlnet_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
)

parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--length", type=int, default=20)
parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)

parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

parser.add_argument("--dataset", type=str)


args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

set_seed(args)

# Initialize the model and tokenizer
try:
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
model = model_class.from_pretrained(args.model_name_or_path)
model.to(args.device)

args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
logger.info(args)

prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

# Different models need different input formatting and/or extra arguments
requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
if requires_preprocessing:
    prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
    preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
    encoded_prompt = tokenizer.encode(
        preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
    )
else:
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
encoded_prompt = encoded_prompt.to(args.device)

output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=args.length + len(encoded_prompt[0]),
    temperature=args.temperature,
    top_k=args.k,
    top_p=args.p,
    repetition_penalty=args.repetition_penalty,
    do_sample=True,
    num_return_sequences=args.num_return_sequences,
)

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(args.stop_token) if args.stop_token else None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    )

    generated_sequences.append(total_sequence)
    print(total_sequence)



import dataloader

assert args.dataset == "MNLI"

sentences = []
with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{args.dataset}/processed/dev_matched.raw.input0", "r") as inFile_premise:
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{args.dataset}/processed/dev_matched.raw.input1", "r") as inFile_hypothesis:
  try:
   while True:
    premise = next(inFile_premise).strip()
    hypothesis = next(inFile_hypothesis).strip()
    sentences.append((premise, hypothesis))
  except StopIteration:
    pass
import re

#quit()

GENERATION_VOCABULARY_MASK = torch.cuda.FloatTensor([float("-inf") if ("<" in tokenizer.convert_ids_to_tokens(x)) else 0 for x in range(32000)]).view(1, 1, -1)

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{args.dataset}/dev_alternatives_f.tsv", "w") as outFile:
  for prompt_text1, prompt_text2 in sentences[500:]:
    # Different models need different input formatting and/or extra arguments
    if not prompt_text1.endswith("."):
       prompt_text1 = prompt_text1 + "."
    if not prompt_text2.endswith("."):
       prompt_text2 = prompt_text2 + "."
    prompt_text = prompt_text1 + "@ " + prompt_text2

    if prompt_text in originalsProcessed:
       print("DONE")
       continue


    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
        encoded_prompt = tokenizer.encode(
            preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
        )
    else:
        assert False
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)
    
    print(encoded_prompt)
    print(encoded_prompt.size())

    LENGTH_OF_INITIAL_PART =  168
    assert tokenizer.convert_ids_to_tokens(int(encoded_prompt[0][LENGTH_OF_INITIAL_PART-1])) == ">", (LENGTH_OF_INITIAL_PART, ([tokenizer.convert_ids_to_tokens(int(x)) for x in encoded_prompt[0]]))

    tokenized_bpe = [tokenizer.convert_ids_to_tokens(int(x)) for x in encoded_prompt[0][LENGTH_OF_INITIAL_PART:]]
    print(tokenized_bpe)
    separationMarks = [i for i in range(len(tokenized_bpe)) if "@" == tokenized_bpe[i]]
    assert len(separationMarks) >= 1, tokenized_bpe
#    assert False
    print(encoded_prompt)
#    quit()
    encoded = [int(x) for x in list(encoded_prompt[0])]
    del encoded[separationMarks[0]+LENGTH_OF_INITIAL_PART]
    print([tokenizer.convert_ids_to_tokens(x) for x in encoded[LENGTH_OF_INITIAL_PART:]])
    assert tokenizer.convert_ids_to_tokens(encoded[separationMarks[0]+LENGTH_OF_INITIAL_PART-1]).endswith(".")
    #quit()

    i = LENGTH_OF_INITIAL_PART

    print("#####", file=outFile)
    print(prompt_text, file=outFile)
    print(" ".join([str(x) for x in separationMarks]), file=outFile) # record the boundaries between the two questions
    sentLength = len(encoded) - LENGTH_OF_INITIAL_PART
    
    import torch.nn.functional as F
    
    
    assert tokenizer.convert_ids_to_tokens(6) == "<mask>"
    
    import collections
    
    queue = collections.deque()
    
    SAMPLES = 10
    
#    print("NUMBER OF SUBSETS", 2**sentLength)
    print(" ".join([tokenizer.convert_ids_to_tokens(int(x)) for x in encoded[LENGTH_OF_INITIAL_PART:]]), file=outFile)
    tokenizedStrings = [tokenizer.convert_ids_to_tokens(int(x)) for x in encoded[LENGTH_OF_INITIAL_PART:]]

    print("Sentence length", sentLength)
    subsets = set()
    # subsets of size 1
    for i in range(sentLength):
       subsets.add(("0"*i) + "1" + ("0"*(sentLength-i-1)))
    # bigrams
    for i in range(sentLength-1):
       subsets.add(("0"*i) + "11" + ("0"*(sentLength-i-2)))
    # trigrams
    for i in range(sentLength-2):
       subsets.add(("0"*i) + "111" + ("0"*(sentLength-i-3)))
    # fourgrams
    for i in range(sentLength-3):
       subsets.add(("0"*i) + "1111" + ("0"*(sentLength-i-4)))

    # fivegrams
    for i in range(sentLength-4):
       subsets.add(("0"*i) + "11111" + ("0"*(sentLength-i-5)))


    # sixgrams
    for i in range(sentLength-5):
       subsets.add(("0"*i) + "111111" + ("0"*(sentLength-i-6)))


#    quit()


    spanLength = int(sentLength/7)+1
    # spans of length sentLength/7
    for subset in range(1, 2**7-1):
      subset_ = format(subset, "b")
      subset_ = ("0"*(7 - len(subset_))) + subset_
      assert len(subset_) == 7, subset_
   #   print(subset_)
      subset__ = "".join([x*spanLength for x in subset_])
      subset__ = subset__[:sentLength]
      assert len(subset__) == sentLength
      subsets.add(subset__)
#    # any subset in a window around the critical word
#    criticalWord = int(sentLength/2)
#    criticalWindowLength = min(8, sentLength)
#    for subset in range(1, 2**criticalWindowLength-1):
#      subset_ = format(subset, "b")
#      subset_ = ("0"*(criticalWindowLength - len(subset_))) + subset_
#      subset_ = ("0"*int((sentLength-criticalWindowLength)/2)) + subset_ + ("0"*int((sentLength-criticalWindowLength)/2))
#      if len(subset_) == sentLength-1:
#        subset_ = subset_ + "0"
#      assert len(subset_) == sentLength, (len(subset_), sentLength)
#      subsets.add(subset_)

    # Make sure the partitionings respect word boundaries (reduces number of subsets and seems linguistically the right thing to do)
  #    print(subset__)
 #   quit()



    # Make sure all subsets respect word boundaries
    subsets_ = set()
    for subset in subsets:
        subset = list(subset)
        print("=====")
        print("BEFORE","".join( subset))
        lastStart = 0
        print([tokenizedStrings[i] if subset[i] == "0" else "XXX" for i in range(len(subset))])
        subset[-1] = "0"
        for i in range(1, len(subset)-1):
             if tokenizedStrings[i].startswith("▁"):
                 lastStart = i
             if subset[i] == "1":
                 if subset[i-1] == "0":
                    if not  tokenizedStrings[i].startswith("▁"):
                        for j in range(lastStart, i):
                           subset[j] = "1"
                 if i+2 < len(subset) and subset[i+1] == "0":
                    if not  tokenizedStrings[i+1].startswith("▁"):
                        for j in range(lastStart, i+1):
                           subset[j] = "0"
        print("AFTER ", "".join(subset))
        print([tokenizedStrings[i] if subset[i] == "0" else "XXX" for i in range(len(subset))])
        assert len(subset) == sentLength, (len(subset), sentLength)

        
        subsets_.add("".join(subset))
    subsets = subsets_
#    print(subsets)
    print(len(subsets))

#    quit()


    subsets_ = set()
    for subset in subsets:
        subset = list(subset)
        subset[-1] = "0"
        for i in separationMarks:
          subset[i-1] = "0"
          assert i > 0
          assert i < len(subset)
        if "1" not in subset:
          continue
        subsets_.add("".join(subset))
    subsets = subsets_

 #   print(subsets)
#    quit()
    print("NUMBER OF SUBSETS", len(subsets))
  #  quit()
    for subset_ in subsets:
    #  print(subset_)
      if "1" not in subset_:
         continue
      encoded_ = torch.LongTensor(encoded).view(1, -1).cuda()
      for i in range(sentLength):
         if subset_[i] == "1":
            encoded_[0,i+LENGTH_OF_INITIAL_PART] = 6
     # print(subset_, [tokenizer.convert_ids_to_tokens(int(x)) for x in encoded[0][LENGTH_OF_INITIAL_PART:]])
      for sample_ in range(SAMPLES):
        assert "1" in subset_

        queue.append({"free" : list(subset_), "subset" : subset_, "encoded" : encoded_.clone(), "sample_id" : sample_})
      # now generate
    
    finished = []
    
    perm_masks = []
    target_mappings = []
    input_idss = []
    points = []
    
    BATCH_SIZE = 128
    
    while len(queue) > 0:
      point = queue.popleft()
      input_ids = point["encoded"]
      subset__ = point["free"]
      assert "1" in subset__
      if True:
        firstMask = subset__.index("1")+LENGTH_OF_INITIAL_PART
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float).to(args.device)
        # Two possibilities. Somehow the second one leads to weirdly random samples
    #    perm_mask[:, :, i] = 1.0  # Previous tokens don't see last token
        for i in range(len(subset__)):
           if subset__[i] == "1":
              perm_mask[:, :, i+LENGTH_OF_INITIAL_PART] = 1.0  # Previous tokens don't see last token
        perm_masks.append(perm_mask)
        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float).to(args.device)  # Shape [1, 1, seq_length] => let's predict one token
        target_mapping[0, 0, firstMask] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        target_mappings.append(target_mapping)
        input_idss.append(input_ids)
        points.append(point)
     #   print(perm_mask.size(), target_mapping.size(), input_ids.size())
    #    quit()
        assert len(points) == (len(perm_masks))
    #    print(len(points))
        if len(points) == BATCH_SIZE or len(queue) == 0:
            perm_mask = torch.cat(perm_masks, dim=0)
            input_ids = torch.cat(input_idss, dim=0)
            target_mapping = torch.cat(target_mappings, dim=0)
                  
            with torch.no_grad():
              outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
              next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        #      print(tokenizer.convert_ids_to_tokens(int((next_token_logits).argmax())))
#              print([tokenizer.convert_ids_to_tokens(int(x)) for x in range(10)]) # ['<unk>', '<s>', '</s>', '<cls>', '<sep>', '<pad>', '<mask>', '<eod>', '<eop>', '.']
 #             print(GENERATION_VOCABULARY_MASK)
  #            print(next_token_logits.size())
   #           quit()    
              probs = F.softmax(next_token_logits + GENERATION_VOCABULARY_MASK, dim=-1).squeeze(1)
        #      print(probs.size())
              next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        #      print(next_token.size())
              for batch in range(len(points)):
       #          print(tokenizer.convert_ids_to_tokens(int((next_token[batch]))))
                 firstMask = points[batch]["free"].index("1")+LENGTH_OF_INITIAL_PART
      #           print("".join(subset__), [tokenizer.convert_ids_to_tokens(int(x)) for x in points[batch]["encoded"][0][LENGTH_OF_INITIAL_PART:]])
                 points[batch]["encoded"][0, firstMask] = int(next_token[batch])
                 points[batch]["free"][firstMask-LENGTH_OF_INITIAL_PART] = "0"
     #            print("".join(subset__), [tokenizer.convert_ids_to_tokens(int(x)) for x in points[batch]["encoded"][0][LENGTH_OF_INITIAL_PART:]])
           
                 if "1" in points[batch]["free"]:
                      queue.append(points[batch])
    #                  print("QUEUE", len(queue))
                      assert "1" in points[batch]["free"]
                 else:
                      finished.append(points[batch])
    #                  print("FINISHED", points[batch])
                      print("FINISHED", "".join(points[batch]["subset"]), [tokenizer.convert_ids_to_tokens(int(x)) for x in points[batch]["encoded"][0][LENGTH_OF_INITIAL_PART:]], len(finished)/(len(queue)+len(finished)))
                      print("".join(points[batch]["subset"]), "\t", " ".join([tokenizer.convert_ids_to_tokens(int(x)) for x in points[batch]["encoded"][0][LENGTH_OF_INITIAL_PART:]]), file=outFile)
            perm_masks = []
            target_mappings = []
            input_idss = []
            points = []
    
    
    

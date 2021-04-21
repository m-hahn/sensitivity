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


# Based on https://raw.githubusercontent.com/huggingface/transformers/master/examples/run_generation.py

# Construct neighbors for MRPC

# ~/python-py37-mhahn generate20_c_OnlySubsetsNoAlternatives.py --model_type="XLNet" --model_name_or_path=xlnet-large-cased --dataset=MRPC

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



assert args.dataset == "MRPC"

#####################################################################
# Collect sentences to construct alternatives for
sentences = []
with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{args.dataset}/dev.tsv", "r") as inFile:
  next(inFile) # for the header
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 5:
       continue
     #assert len(line) == 4
     sentences.append((line[3], line[4]))
######################################################################

import re


# Mask for generating, avoiding generation of undesired tokens

# Open file into which to write alternatives
with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{args.dataset}/dev_alternatives_c_OnlySubsetsNoAlternatives.tsv", "w", encoding="utf-8") as outFile:
  for prompt_text1, prompt_text2 in sentences:
    # Different models need different input formatting and/or extra arguments
    if not prompt_text1.endswith("."):
       prompt_text1 = prompt_text1 + "."
    if not prompt_text2.endswith("."):
       prompt_text2 = prompt_text2 + "."
    prompt_text = prompt_text1 + "@ " + prompt_text2 + " </s>"


    prompt_text = re.sub(r" \'s ", "\'s ", prompt_text)
    prompt_text = re.sub(r" \'m ", "\'m ", prompt_text)
    prompt_text = re.sub(r" \'ve ", "\'ve ", prompt_text)
    prompt_text = re.sub(r" n\'t ", "n\'t ", prompt_text)
    prompt_text = re.sub(r" n \'t ", "n\'t ", prompt_text)
    prompt_text = re.sub(r"n \'t ", "n\'t ", prompt_text)
    prompt_text = re.sub(r" \'re ", "\'re ", prompt_text)
    prompt_text = re.sub(r" \'d ", "\'d ", prompt_text)
    prompt_text = re.sub(r" \'ll", "\'ll", prompt_text)
    prompt_text = re.sub(r" , ", ", ", prompt_text)
    prompt_text = re.sub(r" \.", ".", prompt_text)
    prompt_text = re.sub(r" ! ", "! ", prompt_text)
    prompt_text = re.sub(r"\\", "", prompt_text)
    prompt_text = re.sub(r"\( ", "(", prompt_text)
    prompt_text = re.sub(r" \)", ")", prompt_text)
    prompt_text = re.sub(r" \?", "?", prompt_text)
    prompt_text = re.sub(r"\s{2,}", " ", prompt_text)


    # undo tokenizatiion


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
    print("NUMBER OF SUBSETS", len(subsets))
  #  quit()
    for subset_ in subsets:
    #  print(subset_)
      if "1" not in subset_:
         continue
      encoded_ = torch.LongTensor(encoded).view(1, -1)
      for i in range(sentLength):
         if subset_[i] == "1":
            encoded_[0,i+LENGTH_OF_INITIAL_PART] = 6
     # print(subset_, [tokenizer.convert_ids_to_tokens(int(x)) for x in encoded[0][LENGTH_OF_INITIAL_PART:]])
      for sample_ in range(SAMPLES):
        assert "1" in subset_
        datapoint = ({"free" : list(subset_), "subset" : subset_, "encoded" : encoded_.clone(), "sample_id" : sample_})
        print("".join(datapoint["subset"]), "\t", "__NOTHING__", file=outFile)
   
    

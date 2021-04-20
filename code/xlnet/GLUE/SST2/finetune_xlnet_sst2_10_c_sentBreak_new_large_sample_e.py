# In part based on ~/scr/CODE/transformers/examples/run_glue.py

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

# Construct neighbors for SST-2

# ~/python-py37-mhahn finetune_xlnet_sst2_10_c_sentBreak_new_large_sample_e.py --model_type="XLNet" --model_name_or_path=xlnet-large-cased --dataset=SST2

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


#import dataloader

assert args.dataset == "SST2"

import random

sentences = []
with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/train.tsv", "r") as inFile:
  header = next(inFile) # for the header
  assert header == "sentence\tlabel\n"
#  next(inFile)
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 2:
       continue
     assert len(line) == 2
     sentences.append(line[0])
random.shuffle(sentences)

sentences_dev = []
with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev.tsv", "r") as inFile:
  header = next(inFile) # for the header
  assert header == "sentence\tlabel\n"
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 2:
       continue
     assert len(line) == 2
     sentences_dev.append(line[0])


import re

#quit()




import argparse
import glob
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

LEARNING_RATE=5e-5
WARMUP_STEPS=10
TRAINING_STEPS=1000


LEARNING_RATE = 1e-5 #random.choice([1e-5, 5e-5, 1e-4])
WARMUP_STEPS = 100 #random.choice([10, 100, 1000])
TRAINING_STEPS = 100000 # random.choice([1000, 10000, 100000])

LEARNING_RATE = 1e-5 #random.choice([5e-6, 1e-5, 5e-5])
WARMUP_STEPS = 100 # random.choice([10, 100, 1000])
TRAINING_STEPS = 100000 #random.choice([100000, 1000000, 10000000])


training_args = {}
training_args["WARMUP_STEPS"] = WARMUP_STEPS
training_args["TRAINING_STEPS"] = TRAINING_STEPS
training_args["LEARNING_RATE"] = LEARNING_RATE
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TRAINING_STEPS
)


global_step = 0
epochs_trained = 0
steps_trained_in_current_epoch = 0
tr_loss, logging_loss = 0.0, 0.0
model.zero_grad()



def runOnCorpus(sentences, doTraining = False):
 input_idss = []
 per_masks = []
 target_mappings = []

 losses = []
 for prompt_num, prompt_text in enumerate(sentences):
    # Different models need different input formatting and/or extra arguments
    prompt_text = prompt_text + " </s>"
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
    
    #print(encoded_prompt)
    #print(encoded_prompt.size())

    LENGTH_OF_INITIAL_PART =  168
    assert tokenizer.convert_ids_to_tokens(int(encoded_prompt[0][LENGTH_OF_INITIAL_PART-1])) == ">", (LENGTH_OF_INITIAL_PART, ([tokenizer.convert_ids_to_tokens(int(x)) for x in encoded_prompt[0]]))

    #print([tokenizer.convert_ids_to_tokens(int(x)) for x in encoded_prompt[0][LENGTH_OF_INITIAL_PART:]])
#    assert False

    encoded = [int(x) for x in list(encoded_prompt[0])]
    i = LENGTH_OF_INITIAL_PART

    sentLength = len(encoded) - LENGTH_OF_INITIAL_PART
    
    import torch.nn.functional as F
    
    
    assert tokenizer.convert_ids_to_tokens(6) == "<mask>"
    
    import collections
    
    queue = collections.deque()
    
    SAMPLES = 10
    
    input_ids = encoded
    tokenizedStrings = [tokenizer.convert_ids_to_tokens(int(x)) for x in encoded[LENGTH_OF_INITIAL_PART:]]
    

    textLength = encoded.size()[1]
#    input_ids = torch.cat([encoded, torch.FloatTensor([6 for _ in range(220-textLength)]).cuda().view(1, -1)], dim=1)
    #print(encoded)
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float).to(args.device)
    
    target_mapping = torch.zeros((1, encoded.size()[1]-LENGTH_OF_INITIAL_PART, input_ids.shape[1]), dtype=torch.float).to(args.device)  # Shape [1, 1, seq_length] => let's predict one token
    order = list(range(LENGTH_OF_INITIAL_PART, textLength))
    random.shuffle(order)
    perm_mask[0, :LENGTH_OF_INITIAL_PART, LENGTH_OF_INITIAL_PART:] = 1 # Initial part cannot attend to the future
    for i in range(len(order)): # Make masking follow the specified ordering
       perm_mask[0, order[i], order[i:]] = 1.0
    #masks_strings = sorted(["".join(str(int(float(y))) for  y in x) for x in perm_mask[0]])
    #for x in masks_strings:
   #   print(x[LENGTH_OF_INITIAL_PART-10:])
  #  quit()
    
    #print(perm_mask[:, LENGTH_OF_INITIAL_PART-10:, LENGTH_OF_INITIAL_PART-10:])
    #quit()
    for i in range(LENGTH_OF_INITIAL_PART, textLength):
#       perm_mask[0, :i+1, i] = 1.0
       
       target_mapping[0, i-LENGTH_OF_INITIAL_PART, i] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    input_idss.append(input_ids)
    per_masks.append(perm_mask)
    target_mappings.append(target_mapping)
    assert len(per_masks) == len(target_mappings)

    if doTraining:
       BATCH_SIZE = 2
    else:
       BATCH_SIZE=32

    if len(input_idss) == BATCH_SIZE:
       model.train()
   
       maxInputLength = max([x.size()[1] for x in input_idss])
 #      print(maxInputLength)
       for i in range(len(input_idss)):
          input_idss[i] = torch.cat([input_idss[i], torch.LongTensor([6 for _ in range(maxInputLength-input_idss[i].size()[1])]).cuda().view(1,-1)], dim=1)
  #        print(input_idss[i].size())
       input_ids = torch.cat(input_idss, dim=0)

       for i in range(len(per_masks)):
          per_masks[i] = torch.cat([per_masks[i], 1.0 + torch.zeros(1, maxInputLength - per_masks[i].size()[1], per_masks[i].size()[2]).cuda()], dim=1)
          per_masks[i] = torch.cat([per_masks[i], 1.0 + torch.zeros(1, per_masks[i].size()[1], maxInputLength - per_masks[i].size()[2]).cuda()], dim=2)
   #       print(per_masks[i].size())
       per_mask = torch.cat(per_masks, dim=0)

       for i in range(len(target_mappings)):
          target_mappings[i] = torch.cat([target_mappings[i], torch.zeros(1, maxInputLength - LENGTH_OF_INITIAL_PART - target_mappings[i].size()[1], target_mappings[i].size()[2]).cuda()], dim=1)
          target_mappings[i] = torch.cat([target_mappings[i], torch.zeros(1, maxInputLength - LENGTH_OF_INITIAL_PART, maxInputLength - target_mappings[i].size()[2]).cuda()], dim=2)
    #      print(target_mappings[i].size())
       target_mapping = torch.cat(target_mappings, dim=0)
#       quit()      
       #print([x.size() for x in per_masks], [x.size() for x in target_mappings], len(per_masks), len(target_mappings))
       #assert len(per_masks) == len(target_mappings)

       #print(input_ids.size(), per_mask.size(), target_mapping.size(), maxInputLength)
       #print(per_mask.size())
       #print(target_mapping.size())
       
       #quit()
       if doTraining:
            outputs = model(input_ids, perm_mask=per_mask, target_mapping=target_mapping)
       else:
          with torch.no_grad():
            outputs = model(input_ids, perm_mask=per_mask, target_mapping=target_mapping)
       next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
       #print(next_token_logits.size(), input_ids.size())
     
       ceLoss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=6)(next_token_logits.view(-1, 32000), input_ids[:, LENGTH_OF_INITIAL_PART:].contiguous().view(-1))
      # print(ceLoss)
       loss = ceLoss.mean()
   
       losses.append(float(loss))   
       print(devLosses)
       print(float(loss), prompt_num/len(sentences), sum(losses)/len(losses), doTraining, training_args)
       if doTraining:
          model.zero_grad()
     
          loss.backward()
      
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
          optimizer.step()
          scheduler.step()  # Update learning rate schedule

       input_idss = []
       per_masks = []
       target_mappings = []
 return sum(losses)/len(losses)

devLosses = []

MYID = random.randint(1000, 100000000)

devLosses.append(runOnCorpus(sentences_dev, doTraining = False))
for _ in range(1):
   runOnCorpus(sentences[:200000], doTraining = True)
   devLosses.append(runOnCorpus(sentences_dev, doTraining = False))
   assert devLosses[1] < devLosses[0]
   with open(f"output/{__file__}_{MYID}.txt", "w") as outFile:
       print(devLosses, file=outFile)
       print(training_args, file=outFile)
   if devLosses[-1] > devLosses[-2]:
      break
print("FINISHED FINE-TUNING XLNET", devLosses)
#quit()






#import dataloader

assert args.dataset == "SST2"

sentences = []
with open("/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev.tsv", "r") as inFile:
  header = next(inFile) # for the header
  assert header == "sentence\tlabel\n"
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 2:
       continue
     assert len(line) == 2
     sentences.append(line[0])

import re

#quit()

GENERATION_VOCABULARY_MASK = torch.cuda.FloatTensor([float("-inf") if ("<" in tokenizer.convert_ids_to_tokens(x)) else 0 for x in range(32000)]).view(1, 1, -1)

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_c_sentBreak_new_finetuned_large_e.tsv", "w") as outFile:
  for prompt_text in sentences[400:]:
    prompt_text_original = prompt_text
    prompt_text = prompt_text + " </s>"
    # Different models need different input formatting and/or extra arguments
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

    print([tokenizer.convert_ids_to_tokens(int(x)) for x in encoded_prompt[0][LENGTH_OF_INITIAL_PART:]])
#    assert False

    encoded = [int(x) for x in list(encoded_prompt[0])]
    i = LENGTH_OF_INITIAL_PART

    print("#####", file=outFile)
    print(prompt_text, file=outFile)
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

    # 7grams
    for i in range(sentLength-6):
       subsets.add(("0"*i) + "1111111" + ("0"*(sentLength-i-7)))

    # 8grams
    for i in range(sentLength-7):
       subsets.add(("0"*i) + "11111111" + ("0"*(sentLength-i-8)))


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
        subset[-1] = "0" # for </s>
        if tokenizedStrings[-2] == ".":
            subset[-2] = "0" # for punctuation
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
    
    BATCH_SIZE = 64
    
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
              perm_mask[:, :, i+LENGTH_OF_INITIAL_PART] = 1.0  # No token is allowed to see this one
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
    
    
    

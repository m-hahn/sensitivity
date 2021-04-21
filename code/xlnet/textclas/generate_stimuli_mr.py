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

# Construct neighbors for Text Classification

# ~/python-py37-mhahn generate8_b2.py --model_type="XLNet" --model_name_or_path=xlnet-large-cased --dataset=cr

import argparse
import logging

import numpy as np
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

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

import json

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

parser.add_argument("--dataset", type=str)


args = parser.parse_args()



import dataloader

args.path = "/u/scr/mhahn/software/sent-conv-torch/data"
# https://github.com/asappresearch/sru/blob/master/classification/train_classifier.py
if args.dataset == 'mr':
    data, label = dataloader.read_MR(args.path, clean=False)
elif args.dataset == 'subj':
    data, label = dataloader.read_SUBJ(args.path)
elif args.dataset == 'cr':
    data, label = dataloader.read_CR(args.path)
elif args.dataset == 'mpqa':
    data, label = dataloader.read_MPQA(args.path)
elif args.dataset == 'trec':
    train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path)
    data = train_x + test_x
    label = None
elif args.dataset == 'sst':
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
    data = train_x + valid_x + test_x
    label = None
else:
    raise Exception("unknown dataset: {}".format(args.dataset))

import re

DEV_END = int(len(data)/10)
print(DEV_END)
#quit()


sentences = []

with open("../../experiments/9-mr/expt-files/positiveReviews.js", "w") as outFilePositive:
 with open("../../experiments/9-mr/expt-files/negativeReviews.js", "w") as outFileNegative:
  with open("../../experiments/9-mr/expt-files/reviews.js", "w") as outFileBoth:
   for sentence, label_ in list(zip(data, label))[:DEV_END]:
    string = " ".join(sentence)
    if not (string.endswith(".") or string.endswith("?") or string.endswith("!")):
      string = string+"."
    string = string[0].upper() + string[1:]
   
#    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r" \'s ", "\'s ", string)
    string = re.sub(r" \'m ", "\'m ", string)
    string = re.sub(r" \'ve ", "\'ve ", string)
    string = re.sub(r" n\'t ", "n\'t ", string)
    string = re.sub(r" n \'t ", "n\'t ", string)
    string = re.sub(r"n \'t ", "n\'t ", string)
    string = re.sub(r" \'re ", "\'re ", string)
    string = re.sub(r" \'d ", "\'d ", string)
    string = re.sub(r" \'ll", "\'ll", string)
    string = re.sub(r" , ", ", ", string)
    string = re.sub(r" \.", ".", string)
    string = re.sub(r" ! ", "! ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\( ", "(", string)
    string = re.sub(r" \)", ")", string)
    string = re.sub(r" \?", "?", string)
    string = re.sub(r"\s{2,}", " ", string)
    if label_ == 1:
       print(json.dumps({"sentence" : string.strip().split(" ")})+",", file=outFilePositive)
    else:
       print(json.dumps({"sentence" : string.strip().split(" ")})+",", file=outFileNegative)
    print(json.dumps({"sentence" : string.strip().split(" "), "label" : label_})+",", file=outFileBoth)


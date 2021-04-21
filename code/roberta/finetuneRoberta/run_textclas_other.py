# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

# https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""
export GLUE_DIR=/sailhome/mhahn/scr/PRETRAINED/CACHED/
export TASK_NAME=mpqa
rm /sailhome/mhahn/scr/PRETRAINED/CACHED/*/cached_*_Roberta* 
rm -r /tmp/$TASK_NAME
~/python-py37-mhahn run_textclas.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
rm /sailhome/mhahn/scr/PRETRAINED/CACHED/*/cached_*_Roberta* 
rm -r /tmp/mpqa
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
args, _ = parser.parse_known_args()
print(args)

with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.task_name}_alternatives.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

from transformers.data.processors.utils import InputExample

alternatives_examples = []

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   
   alternative = alternative.split("\n")
   print(alternative[:5])
   sentence = alternative[1].replace(" ").replace("▁", " ").replace("</s>", "").strip() # tokenized original
   if sentence not in variants_set:
   #  print(sentence)
     variants_set.add(sentence)
     alternatives_examples.append(InputExample(guid=f"alternatives_{len(alternatives_examples)}", text_a=sentence, label="1"))



   for line in alternative[2:]:
     if len(line) < 5:
       continue
     subset, sentence= line.strip().split("\t")
      
     sentence = sentence.replace(" ").replace("▁", " ").replace("</s>", "").strip()
#     result = [""]
#     for word in sentence.strip().split(" "):
#        if word.startswith("▁"):
#            result.append(word[1:])
#        else:
#            result[-1] = result[-1] + word
#     sentence = " ".join(result)
#     sentence = sentence.replace("  ", " ")
#     sentence = sentence.strip()
     if sentence not in variants_set:
     #  print(sentence)
       variants_set.add(sentence)
       alternatives_examples.append(InputExample(guid=f"alternatives_{len(alternatives_examples)}", text_a=sentence, label="1"))

print(len(alternatives_examples))
#quit()

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from textclas import (TextclasDataset, TextclasDataTrainingArguments)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    #GlueDataset,
    GlueDataTrainingArguments,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

# Set seed
set_seed(training_args.seed)

try:
    num_labels = 2 #glue_tasks_num_labels['cola']
    output_mode = "classification" #glue_output_modes['cola']
    print(num_labels, type(output_mode))
except KeyError:
    raise ValueError("Task not found: %s" % ('cola'))

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task='cola',
    cache_dir=model_args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

# Get datasets
train_dataset = (
    TextclasDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
    if training_args.do_train
    else None
)
eval_dataset = (
    TextclasDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
    if training_args.do_eval
    else None
)

def compute_metrics(p: EvalPrediction) -> Dict:
    if output_mode == "classification":
        preds = np.argmax(p.predictions, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(p.predictions)
    return glue_compute_metrics('cola', preds, p.label_ids)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Training
if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()

# Evaluation
results = {}
if training_args.do_eval and training_args.local_rank in [-1, 0]:
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_datasets = [eval_dataset]

    for eval_dataset in eval_datasets:
        result = trainer.evaluate(eval_dataset=eval_dataset)
 #       predictions = trainer.predict(test_dataset=eval_dataset)
#        print(predictions)
#        quit()
        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        results.update(result)


    alternatives = TextclasDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True, datapoints=alternatives_examples)

    predictions = trainer.predict(test_dataset=alternatives)
    print(predictions)

    import scipy
    predictions_softmax = scipy.special.softmax(predictions.predictions, axis=1)
    averagePrediction = predictions_softmax[:,1] - predictions_softmax[:,0]
    print(averagePrediction)
    predictions = np.argmax(predictions.predictions, axis=1)

    print(predictions)
    outPath = f"/u/scr/mhahn/PRETRAINED/textclas/{args.task_name}_alternatives_Predictions.txt"
    print("PRINTING TO", outPath)
    with open(outPath, "w") as outFile:
       for i in range(len(predictions)):
           print(alternatives_examples[i].text_a, "\t", int(predictions[i]), "\t", float(averagePrediction[i]), file=outFile)

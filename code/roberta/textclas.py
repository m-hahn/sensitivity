import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data.dataset import Dataset

from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from transformers.trainer import torch_distributed_zero_first
from transformers.data.processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from transformers.data.processors.utils import InputFeatures

from transformers.data.processors.utils import InputExample


import re

logger = logging.getLogger(__name__)



import dataloader


@dataclass
class TextclasDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class TextclasDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: TextclasDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    
    def getDataset(self, partition):
       path = "/u/scr/mhahn/software/sent-conv-torch/data"
#       self.args.task_name = "mr" 
#       assert self.args.task_name == "cr"
       # https://github.com/asappresearch/sru/blob/master/classification/train_classifier.py
       if self.args.task_name == 'mr':
           data, label = dataloader.read_MR(path)
       elif self.args.task_name == 'subj':
           data, label = dataloader.read_SUBJ(path)
       elif self.args.task_name == 'cr':
           data, label = dataloader.read_CR(path)
       elif self.args.task_name == 'mpqa':
           data, label = dataloader.read_MPQA(path)
       elif self.args.task_name == 'trec':
           train_x, train_y, test_x, test_y = dataloader.read_TREC(path)
           data = train_x + test_x
           label = None
       elif self.args.task_name == 'sst':
           train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(path)
           data = train_x + valid_x + test_x
           label = None
       else:
           raise Exception("unknown task_name: {}".format(self.args.task_name))
       
       DEV_END = int(len(data)/10)
       TEST_END = 2*int(len(data)/10)
      
       print(len(data), DEV_END, TEST_END)
#       assert False
       if partition == "dev":
         data = data[DEV_END:TEST_END]
         label = label[DEV_END:TEST_END]
       elif partition == "train":   
         data = data[TEST_END:]
         label = label[TEST_END:]
       else:
         assert False
       assert set(label) == set([0.0,1.0])
       examples = []
       for i in range(len(data)):
          sentence = data[i]
          if len(sentence) == 0:
             continue
          # In line with the alternatives, add a final punctuation
          if sentence[-1] not in [".", "?", "!"]:
            sentence.append(".")
          sentence = " ".join(sentence)
          print("AT112", data[i])
          sentence = sentence[0].upper() + sentence[1:]
         
      #    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
          sentence = re.sub(r" \'s ", "\'s ", sentence)
          sentence = re.sub(r" \'m ", "\'m ", sentence)
          sentence = re.sub(r" \'ve ", "\'ve ", sentence)
          sentence = re.sub(r" n\'t ", "n\'t ", sentence)
          sentence = re.sub(r" n \'t ", "n\'t ", sentence)
          sentence = re.sub(r"n \'t ", "n\'t ", sentence)
          sentence = re.sub(r" \'re ", "\'re ", sentence)
          sentence = re.sub(r" \'d ", "\'d ", sentence)
          sentence = re.sub(r" \'ll", "\'ll", sentence)
          sentence = re.sub(r" , ", ", ", sentence)
          sentence = re.sub(r" \.", ".", sentence)
          sentence = re.sub(r" ! ", "! ", sentence)
          sentence = re.sub(r"\\", "", sentence)
          sentence = re.sub(r"\( ", "(", sentence)
          sentence = re.sub(r" \)", ")", sentence)
          sentence = re.sub(r" \?", "?", sentence)
          sentence = re.sub(r"\s{2,}", " ", sentence)
      
          examples.append(InputExample(guid=f"{self.args.task_name}_{partition}_{i}", text_a=sentence, label=str(label[i])))
       #print(examples[:10])
       #assert False
       return examples

    def __init__(
        self,
        args: TextclasDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate=False,
        local_rank=-1,
        datapoints=None
    ):
        self.args = args
        print(args)
        processor = glue_processors["cola"]()
        self.task_name = args.task_name
        self.output_mode = glue_output_modes["cola"]
        print(self.output_mode, type(self.output_mode))
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                ("dev" if datapoints is None else "alternatives") if evaluate else "train", tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                print("label_list",label_list)
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    assert False
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                if datapoints is None:
                   examples = (
                    self.getDataset("dev")
#                    processor.get_dev_examples(args.data_dir)
                    if evaluate
                    else self.getDataset("train")
                  )
                else:
                   examples = datapoints
                print(InputExample(guid="train_1000", text_a="This is a bad movie.", label=1.0))
                print("EXAMPLES", examples[:10])
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                print(self.features[:10])
                if local_rank in [-1, 0]:
                    start = time.time()
                    torch.save(self.features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

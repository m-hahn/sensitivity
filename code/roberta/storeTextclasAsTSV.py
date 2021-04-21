import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args, _ = parser.parse_known_args()
print(args)


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
    label = train_y + test_y
elif args.dataset == 'sst':
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
    data = train_x + valid_x + test_x
    label = None
else:
    raise Exception("unknown dataset: {}".format(args.dataset))

import re

DEV_END = int(len(data)/10)
print(DEV_END)

import re

# Preprocessing is taken from ../xlnet/textclas/finetune_xlnet_textclas_8_d.py

def processSentence(sentence):
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
    return string

import dataloader


label_train = label[DEV_END:]
label_dev = label[:DEV_END]
data_train = [processSentence(x) for x in data[DEV_END:]]
data_dev = [processSentence(x) for x in data[:DEV_END]]

print(data_dev[:10])

assert args.dataset in ["mr", "mpqa","trec", "subj", "cr"]

with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/train.tsv", "w") as outFile:
   print("sentence\tlabel", file=outFile)
   for i in range(len(label_train)):
      print(f"{data_train[i]}\t{label_train[i]}", file=outFile)
with open(f"/u/scr/mhahn/PRETRAINED/textclas/{args.dataset}/dev.tsv", "w") as outFile:
   print("sentence\tlabel", file=outFile)
   for i in range(len(label_dev)):
      print(f"{data_dev[i]}\t{label_dev[i]}", file=outFile)


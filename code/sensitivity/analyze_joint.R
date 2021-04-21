

library(dplyr)
library(tidyr)
library(ggplot2)


qqp = read.csv("lstm-qqp.tsv", sep="\t", quote="#")  %>% mutate(Task = "QQP")
sst2 = read.csv("lstm-sst2.tsv", sep="\t", quote="#")       %>% mutate(Task = "SST2")
rte = read.csv("lstm-rte.tsv", sep="\t", quote="#") %>% mutate(Task = "RTE")


data = rbind(qqp, sst2)
data = rbind(data, rte)

plot = ggplot(data, aes(x=BinaryS1ensitivity, y=LSTM_Sensitivity, color=Task, group=Task))+  geom_point(alpha=0.1)+ geom_smooth(se=F)  + theme_bw() + xlab("RoBERTa Sensitivity") + ylab("BiLSTM Sensitivity") + theme(legend.position = 'bottom') + xlim(NA, 4)
ggsave(plot, file="roberta-lstm-joint.pdf", height=3, width=3)




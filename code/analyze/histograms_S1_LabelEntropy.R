

library(tidyr)
library(dplyr)
library(ggplot2)

bow = read.csv("bow.txt", sep="\t")
entropy = read.csv("label_entropy.txt", sep="\t")
bow = merge(bow, entropy, by=c("Task"), all=TRUE)

bow = bow %>% mutate(CBOWAccuracy = ifelse(is.na(PairCBOWAccuracy), IndivCBOWAccuracy, PairCBOWAccuracy))

bow = bow %>% mutate(CBOWErrorReduction = (CBOWAccuracy-MajorityClass)/(1-MajorityClass))
bow = bow %>% mutate(IndivCBOWErrorReduction = (IndivCBOWAccuracy-MajorityClass)/(1-MajorityClass))

bow = bow %>% mutate(LSTMErrorReduction = pmax(0,(LSTMAccuracy-MajorityClass)/(1-MajorityClass)))
bow = bow %>% mutate(RoBERTaErrorReduction = pmax(0,(RoBERTaAccuracy-MajorityClass)/(1-MajorityClass)))

cor.test(bow$LabelEntropy, bow$IndivCBOWErrorReduction)
cor.test(bow$LabelEntropy, bow$LSTMErrorReduction)


library(ggrepel)


d1 = bow %>% mutate(Model = "BoE", ErrorReduction = IndivCBOWErrorReduction)
d2 = bow %>% mutate(Model = "BiLSTM", ErrorReduction = LSTMErrorReduction)
d3 = bow %>% mutate(Model = "RoBERTa", ErrorReduction = RoBERTaErrorReduction)



bowLong = rbind(d1, d2, d3)

bowLong$Model = as.factor(bowLong$Model, levels=c("BoE", "BiLSTM", "RoBERTa"))

plot = ggplot(data=bowLong, aes(x=LabelEntropy, y=100*ErrorReduction)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none") + facet_grid(~Model) + theme(axis.text = element_text(size=12), axis.title = element_text(size=15), strip.text = element_text(size = 18))
plot = plot + ylab("Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="label-entropy-accuracy-grid.pdf", width=9, height=4)



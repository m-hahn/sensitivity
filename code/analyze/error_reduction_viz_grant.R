

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

byTask = read.csv("xlnet-s1ensitivities.tsv", sep="\t") %>% mutate(Type=NULL)
types = read.csv("types.tsv", sep="\t")
byTask = merge(byTask, types, by=c("Task"))
bow = read.csv("bow.txt", sep="\t")

byTask = merge(byTask, bow, by=c("Task"), all=TRUE)

byTask = byTask %>% mutate(CBOWAccuracy = ifelse(is.na(PairCBOWAccuracy), IndivCBOWAccuracy, PairCBOWAccuracy))

byTask = byTask %>% mutate(CBOWErrorReduction = (CBOWAccuracy-MajorityClass)/(1-MajorityClass))
byTask = byTask %>% mutate(IndivCBOWErrorReduction = (IndivCBOWAccuracy-MajorityClass)/(1-MajorityClass))

byTask = byTask %>% mutate(LSTMErrorReduction = pmax(0,(LSTMAccuracy-MajorityClass)/(1-MajorityClass)))
byTask = byTask %>% mutate(RoBERTaErrorReduction = pmax(0,(RoBERTaAccuracy-MajorityClass)/(1-MajorityClass)))



library(ggrepel)



#plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=CBOWErrorReduction, color=Type)) + geom_point() + geom_label(aes(label=Task)) + theme_bw()


d2 = byTask %>% mutate(Model = "BiLSTM", ErrorReduction = LSTMErrorReduction)
d3 = byTask %>% mutate(Model = "RoBERTa", ErrorReduction = RoBERTaErrorReduction)

byTaskLong = rbind(d2, d3)

byTaskLong$Model = factor(byTaskLong$Model, levels=c("BiLSTM", "RoBERTa"))


byTaskLong$Type = as.character(byTaskLong$Type)

byTaskLong = byTaskLong %>% filter(!is.na(Type))

byTaskLong[byTaskLong$Type == "TextClas",]$Type = "Text Clas."
byTaskLong[byTaskLong$Type == "Gym",]$Type = "Syntax"

byTaskLong = byTaskLong %>% filter(Type != "Parsing")
byTaskLong = byTaskLong %>% filter(Type != "Syntax")

plot = ggplot(data=byTaskLong, aes(x=MeanS1ensitivity, y=100*ErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()   + facet_grid(~Model) + theme(axis.text = element_text(size=12), axis.title = element_text(size=15), strip.text = element_text(size = 18))
plot = plot + ylab("Error Reduction (%)") + xlab("Average Sensitivity") + ylim(0,100) + theme(legend.position="bottom") + scale_color_manual(values = c("red", "blue"))
ggsave(plot, file="s1ensitivity-accuracy-grid_grant.pdf", width=4.8, height=4)



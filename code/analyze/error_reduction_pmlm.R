

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

byTask = read.csv("pmlm_sensitivities.tsv", sep="\t") %>% rename(MeanS1ensitivity=S1ensitivity)
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



plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*IndivCBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-bowAccuracy-pmlm.pdf", width=3, height=4)


plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*LSTMErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("LSTM Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-lstmAccuracy-pmlm.pdf", width=3, height=4)

plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*RoBERTaErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw() + theme(legend.position="none")
plot = plot + ylab("RoBERTa Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-robertaAccuracy-pmlm.pdf", width=3, height=4)


sink("output/corr-s1ensitivity-errorReduction-pmlm.txt")
print(cor.test(byTask$MeanS1ensitivity, byTask$RoBERTaErrorReduction))
print(cor.test(byTask$MeanS1ensitivity, byTask$LSTMErrorReduction))
print(cor.test(byTask$MeanS1ensitivity, byTask$CBOWErrorReduction))
sink()


data = data%>% mutate(BoECanDo = (BinaryS1ensitivity < 4))
BoEGain = data %>% group_by(Task, Type) %>% summarise(BoEPredicted = mean(BoECanDo))

byTask = merge(byTask, BoEGain, by=c("Task", "Type"), all=TRUE)
byTask = byTask %>% mutate(BoEPredictedAccuracy = BoEPredicted + 0.5*(1-BoEPredicted))
byTask = byTask %>% mutate(BoEErrorReduction = pmax(0,(BoEPredictedAccuracy-MajorityClass)/(1-MajorityClass)))


plot = ggplot(data=byTask, aes(y=BoEErrorReduction, x=MeanS1ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")

plot = ggplot(data=byTask, aes(y=BoEErrorReduction, x=CBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")



plot = ggplot(data=byTask, aes(y=BoEPredictedAccuracy, x=MeanS1ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")



plot = ggplot(data=byTask, aes(y=BoEPredicted, x=MeanS1ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")


plot = ggplot(data=byTask, aes(x=BoEPredicted, y=100*IndivCBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")
#ggsave(plot, file="sensitivity-bowAccuracy.pdf", width=3, height=4)





#plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=CBOWErrorReduction, color=Type)) + geom_point() + geom_label(aes(label=Task)) + theme_bw()


d1 = byTask %>% mutate(Model = "BoE", ErrorReduction = IndivCBOWErrorReduction)
d2 = byTask %>% mutate(Model = "BiLSTM", ErrorReduction = LSTMErrorReduction)
d3 = byTask %>% mutate(Model = "RoBERTa", ErrorReduction = RoBERTaErrorReduction)

byTaskLong = rbind(d1, d2, d3)

byTaskLong$Model = as.factor(byTaskLong$Model, levels=c("BoE", "BiLSTM", "RoBERTa"))

plot = ggplot(data=byTaskLong, aes(x=MeanS1ensitivity, y=100*ErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none") + facet_grid(~Model) + theme(axis.text = element_text(size=12), axis.title = element_text(size=15), strip.text = element_text(size = 18))
plot = plot + ylab("Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-accuracy-grid-pmlm.pdf", width=9, height=4)



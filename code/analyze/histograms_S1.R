

library(tidyr)
library(dplyr)
library(ggplot2)


cr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="Text Cl.")
mr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="Text Cl.")
subj_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="Text Cl.")
mpqa_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="Text Cl.")

# GLUE
cola = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_CoLA_Finetuned.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE")
mnli = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_MNLI_c.py", sep="\t", quote="#") %>% mutate(Task = "MNLI", Type="GLUE")
qqp = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_QQP.py", sep="\t", quote="#") %>% mutate(Task = "QQP", Type="GLUE")
rte = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_RTE_c.py", sep="\t", quote="#") %>% mutate(Task = "RTE", Type="GLUE")
wsc = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_WSC.py", sep="\t", quote="#") %>% mutate(Task = "WSC", Type="GLUE")
sts = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_STS-B_c.py", sep="\t", quote="#") %>% mutate(Task = "STS-B", Type="GLUE")
qnli = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_QNLI_c.py", sep="\t", quote="#") %>% mutate(Task = "QNLI", Type="GLUE")
sst2 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2.py", sep="\t", quote="#") %>% mutate(Task = "SST2", Type="GLUE")
mrpc = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_MRPC_c.py", sep="\t", quote="#") %>% mutate(Task = "MRPC", Type="GLUE")

# Parsing
parsing = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_parsing_c.py", sep="\t", quote="#") %>% mutate(Task = "Labels", Type="Parsing")
parsing_position = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_parsing_position_c.py", sep="\t", quote="#") %>% mutate(Task = "Heads", Type="Parsing")
tagging = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_tagging_c.py", sep="\t", quote="#") %>% mutate(Task = "Tagging", Type="Parsing")

# SyntaxGym
# performance of GPT2 is far from perfect on this task
gym248 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SyntaxGym_248_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym248", Type="Syntax")
gym260 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SyntaxGym_260_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym260", Type="Syntax")

	

data = rbind(cr_finetuned, mr_finetuned, subj_finetuned, mpqa_finetuned)
data = rbind(data, cola, mnli, qqp, rte, wsc, sts, qnli, sst2, mrpc)
data = rbind(data, parsing, parsing_position, tagging)
data = rbind(data, gym248, gym260)



library(stringr)
countWhitespace = function(string) {
	return(str_count(string, " "))
}


data$SentLength = Vectorize(countWhitespace)(as.character(data$Original))



#plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
#plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinaryS1ensitivity, group=Type, color=Type)) + geom_point() + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 50), aes(x=SentLength, y=BinaryS1ensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw() + xlab("Length") + ylab("Average Block Sensitivity")
ggsave(plot, file="byLength_s1ensitivity_textclas_glue.pdf", width=2.5, height=2.5)

#plot = ggplot(data=data %>% filter(SentLength < 100, Type == "Text Cl." | Task == "CoLA"), aes(x=SentLength, y=BinaryS1ensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 100, Type == "Text Cl." | Task == "CoLA"), aes(x=SentLength, y=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw() + xlab("Length") + ylab("Average Block Sensitivity")
ggsave(plot, file="byLength_s1ensitivity_textclas_cola.pdf", width=2.5, height=2.5)
#plot = ggplot(data=data %>% filter(SentLength < 100, Task == "CR" | Task == "CoLA"), aes(x=SentLength, y=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)


# a way to force monotonicity
# https://stats.stackexchange.com/questions/197509/how-to-smooth-data-and-force-monotonicity
#library(scam)
#S1ensitivityHat <- predict(scam(BinaryS1ensitivity ~ s(SentLength, bs = "mpi"), data = data %>% filter(Task == "CR")))
#plot((data %>% filter(Task == "CR"))$SentLength, S1ensitivityHat)

data$Task = as.factor(data$Task)
data$Type = as.factor(data$Type)


plot = ggplot(data=data, aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,6.5) + xlab("Block Sensitivity") + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
ggsave(plot, file="histogram_s1_byTask.pdf", height=6, width=2)

plot = ggplot(data=data %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity") + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,6.5) + xlab("Average Block Sensitivity") + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())
ggsave(plot, file="means_s1_byTask.pdf", height=6, width=3)


library(ggpubr)

for(type in c("GLUE", "Parsing", "Syntax", "Text Cl.")) {
   plot1 = ggplot(data=data %>% filter(Type == type), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density(aes(y=scaled)) + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none", text=element_text(size=16))
   plot2 = ggplot(data=data %>% filter(Type == type) %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity", width=0.1) + theme_bw() + xlim(0.5,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none", text=element_text(size=16))
   plot = ggarrange(plot1, plot2, common.legend=TRUE, legend="bottom")
   if(type == "Text Cl.") {
	   type = "textclas"
   }
   ggsave(plot, file=paste("joint_", type, ".pdf", sep=""), height=2, width=7)
}

plot1 = ggplot(data=data %>% filter(Type == "Parsing"), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density(aes(y=..scaled..)) + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
plot2 = ggplot(data=data %>% filter(Type == "Parsing") %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity", width=0.1) + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
plot = ggarrange(plot1, plot2, common.legend=TRUE, legend="bottom")
#
#plot1 = ggplot(data=data %>% filter(Type == "Parsing"), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot2 = ggplot(data=data %>% filter(Type == "Parsing") %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity") + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot = ggarrange(plot1, plot2, common.legend=TRUE, legend="bottom")
#
#
#plot1 = ggplot(data=data %>% filter(Type == "Syntax"), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot2 = ggplot(data=data %>% filter(Type == "Syntax") %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity") + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot = ggarrange(plot1, plot2, common.legend=TRUE, legend="bottom")
#
#
#
#plot1 = ggplot(data=data %>% filter(Type == "Text Cl."), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot2 = ggplot(data=data %>% filter(Type == "Text Cl.") %>% group_by(Task, Type) %>% summarize(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE)), aes(x=BinaryS1ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity") + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
#plot = ggarrange(plot1, plot2, common.legend=TRUE, legend="bottom")
#
#


#plot = ggplot(data=data, aes(x=BinaryS1ensitivity/SentLength, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,1) + xlab("Block S1ensitivity / Length")



#plot = ggplot(data=data %>% filter(Task=="Subj"), aes(x=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="CR"), aes(x=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="MPQA"), aes(x=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinaryS1ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()



byTask = data %>% group_by(Task, Type) %>% summarise(MeanS1ensitivity = mean(BinaryS1ensitivity), MedianS1ensitivity = median(BinaryS1ensitivity), SES1ensitivity = sd(BinaryS1ensitivity)/sqrt(NROW(BinaryS1ensitivity)))
byTask[order(byTask$MeanS1ensitivity),]

bow = read.csv("bow.txt", sep="\t")

byTask = merge(byTask, bow, by=c("Task"), all=TRUE)

byTask = byTask %>% mutate(CBOWAccuracy = ifelse(is.na(PairCBOWAccuracy), IndivCBOWAccuracy, PairCBOWAccuracy))

byTask = byTask %>% mutate(CBOWErrorReduction = (CBOWAccuracy-MajorityClass)/(1-MajorityClass))
byTask = byTask %>% mutate(IndivCBOWErrorReduction = (IndivCBOWAccuracy-MajorityClass)/(1-MajorityClass))

byTask = byTask %>% mutate(LSTMErrorReduction = pmax(0,(LSTMAccuracy-MajorityClass)/(1-MajorityClass)))
byTask = byTask %>% mutate(RoBERTaErrorReduction = pmax(0,(RoBERTaAccuracy-MajorityClass)/(1-MajorityClass)))



library(ggrepel)

plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*CBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw() 
plot = plot + ylab("CBOW Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
#ggsave(plot, file="sensitivity-bowAccuracy.pdf", width=4, height=4)



plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*IndivCBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-bowAccuracy.pdf", width=3, height=4)


plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*LSTMErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("LSTM Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-lstmAccuracy.pdf", width=3, height=4)

plot = ggplot(data=byTask, aes(x=MeanS1ensitivity, y=100*RoBERTaErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw() + theme(legend.position="none")
plot = plot + ylab("RoBERTa Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s1ensitivity-robertaAccuracy.pdf", width=3, height=4)


sink("output/corr-s1ensitivity-errorReduction.txt")
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
ggsave(plot, file="s1ensitivity-accuracy-grid.pdf", width=9, height=4)



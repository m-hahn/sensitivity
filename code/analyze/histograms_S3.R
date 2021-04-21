

library(tidyr)
library(dplyr)
library(ggplot2)


cr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_textclas.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="textclas")
mr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_textclas.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="textclas")
subj_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_textclas.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="textclas")
mpqa_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_textclas.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="textclas")

# GLUE
cola = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_CoLA_Finetuned.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE")
mnli = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_MNLI_c.py", sep="\t", quote="#") %>% mutate(Task = "MNLI", Type="GLUE")
qqp = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_QQP.py", sep="\t", quote="#") %>% mutate(Task = "QQP", Type="GLUE")
rte = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_RTE_c.py", sep="\t", quote="#") %>% mutate(Task = "RTE", Type="GLUE")
wsc = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_WSC.py", sep="\t", quote="#") %>% mutate(Task = "WSC", Type="GLUE")
sts = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_STS-B_c.py", sep="\t", quote="#") %>% mutate(Task = "STS-B", Type="GLUE")
qnli = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_QNLI_c.py", sep="\t", quote="#") %>% mutate(Task = "QNLI", Type="GLUE")
sst2 = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_SST2.py", sep="\t", quote="#") %>% mutate(Task = "SST2", Type="GLUE")
mrpc = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_MRPC_c.py", sep="\t", quote="#") %>% mutate(Task = "MRPC", Type="GLUE")

# Parsing
parsing = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_parsing_c.py", sep="\t", quote="#") %>% mutate(Task = "Labels", Type="parsing")
parsing_position = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_parsing_position_c.py", sep="\t", quote="#") %>% mutate(Task = "Heads", Type="parsing")
tagging = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_tagging_c.py", sep="\t", quote="#") %>% mutate(Task = "Tagging", Type="parsing")

# SyntaxGym
# performance of GPT2 is far from perfect on this task
gym248 = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_SyntaxGym_248_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym248", Type="Syntax")
gym260 = read.csv("~/CS_SCR/sensitivity/sensitivities/s3ensitivities_estimateS3ensitivity_SyntaxGym_260_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym260", Type="Syntax")

	

data = rbind(cr_finetuned, mr_finetuned, subj_finetuned, mpqa_finetuned)
data = rbind(data, cola, mnli, qqp, rte, wsc, sts, qnli, sst2, mrpc)
data = rbind(data, parsing, parsing_position, tagging)
data = rbind(data, gym248, gym260)



library(stringr)
countWhitespace = function(string) {
	return(str_count(string, " "))
}


data$SentLength = Vectorize(countWhitespace)(as.character(data$Original))



#plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
#plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinaryS3ensitivity, group=Type, color=Type)) + geom_point() + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 50), aes(x=SentLength, y=BinaryS3ensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw() + xlab("Length") + ylab("Average Block Sensitivity")
ggsave(plot, file="byLength_s3ensitivity_textclas_glue.pdf", width=3, height=3)

#plot = ggplot(data=data %>% filter(SentLength < 100, Type == "textclas" | Task == "CoLA"), aes(x=SentLength, y=BinaryS3ensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 100, Type == "textclas" | Task == "CoLA"), aes(x=SentLength, y=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw() + xlab("Length") + ylab("Average Block Sensitivity")
ggsave(plot, file="byLength_s3ensitivity_textclas_cola.pdf", width=3, height=3)
#plot = ggplot(data=data %>% filter(SentLength < 100, Task == "CR" | Task == "CoLA"), aes(x=SentLength, y=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)


# a way to force monotonicity
# https://stats.stackexchange.com/questions/197509/how-to-smooth-data-and-force-monotonicity
#library(scam)
#S3ensitivityHat <- predict(scam(BinaryS3ensitivity ~ s(SentLength, bs = "mpi"), data = data %>% filter(Task == "CR")))
#plot((data %>% filter(Task == "CR"))$SentLength, S3ensitivityHat)

data$Task = as.factor(data$Task)
data$Type = as.factor(data$Type)


plot = ggplot(data=data, aes(x=BinaryS3ensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,12) + xlab("Block Sensitivity") + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), legend.position="none")
ggsave(plot, file="histogram_s3_byTask.pdf", height=6, width=2)

plot = ggplot(data=data %>% group_by(Task, Type) %>% summarize(BinaryS3ensitivity=mean(BinaryS3ensitivity, na.rm=TRUE)), aes(x=BinaryS3ensitivity, y=1, group=Task, color=Task, fill=Task)) + geom_bar(stat="identity") + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,12) + xlab("Average Block Sensitivity") + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())
ggsave(plot, file="means_s3_byTask.pdf", height=6, width=3)

#plot = ggplot(data=data, aes(x=BinaryS3ensitivity/SentLength, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,1) + xlab("Block S3ensitivity / Length")



#plot = ggplot(data=data %>% filter(Task=="Subj"), aes(x=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="CR"), aes(x=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="MPQA"), aes(x=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
#plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinaryS3ensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()



byTask = data %>% group_by(Task, Type) %>% summarise(MeanS3ensitivity = mean(BinaryS3ensitivity), MedianS3ensitivity = median(BinaryS3ensitivity), SES3ensitivity = sd(BinaryS3ensitivity)/sqrt(NROW(BinaryS3ensitivity)))
byTask[order(byTask$MeanS3ensitivity),]

bow = read.csv("bow.txt", sep="\t")

byTask = merge(byTask, bow, by=c("Task"), all=TRUE)

byTask = byTask %>% mutate(CBOWAccuracy = ifelse(is.na(PairCBOWAccuracy), IndivCBOWAccuracy, PairCBOWAccuracy))

byTask = byTask %>% mutate(CBOWErrorReduction = (CBOWAccuracy-MajorityClass)/(1-MajorityClass))
byTask = byTask %>% mutate(IndivCBOWErrorReduction = (IndivCBOWAccuracy-MajorityClass)/(1-MajorityClass))

byTask = byTask %>% mutate(LSTMErrorReduction = pmax(0,(LSTMAccuracy-MajorityClass)/(1-MajorityClass)))
byTask = byTask %>% mutate(RoBERTaErrorReduction = pmax(0,(RoBERTaAccuracy-MajorityClass)/(1-MajorityClass)))



library(ggrepel)

plot = ggplot(data=byTask, aes(x=MeanS3ensitivity, y=100*CBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw() 
plot = plot + ylab("CBOW Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
#ggsave(plot, file="sensitivity-bowAccuracy.pdf", width=4, height=4)



plot = ggplot(data=byTask, aes(x=MeanS3ensitivity, y=100*IndivCBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s3ensitivity-bowAccuracy.pdf", width=3, height=4)


plot = ggplot(data=byTask, aes(x=MeanS3ensitivity, y=100*LSTMErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("LSTM Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s3ensitivity-lstmAccuracy.pdf", width=3, height=4)

plot = ggplot(data=byTask, aes(x=MeanS3ensitivity, y=100*RoBERTaErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw() + theme(legend.position="none")
plot = plot + ylab("RoBERTa Error Reduction (%)") + xlab("Average Block Sensitivity") + ylim(0,100)
ggsave(plot, file="s3ensitivity-robertaAccuracy.pdf", width=3, height=4)


sink("output/corr-s3ensitivity-errorReduction.txt")
cor.test(byTask$MeanS3ensitivity, byTask$RoBERTaErrorReduction)
cor.test(byTask$MeanS3ensitivity, byTask$LSTMErrorReduction)
cor.test(byTask$MeanS3ensitivity, byTask$CBOWErrorReduction)
sink()


data = data%>% mutate(BoECanDo = (BinaryS3ensitivity < 4))
BoEGain = data %>% group_by(Task, Type) %>% summarise(BoEPredicted = mean(BoECanDo))

byTask = merge(byTask, BoEGain, by=c("Task", "Type"), all=TRUE)
byTask = byTask %>% mutate(BoEPredictedAccuracy = BoEPredicted + 0.5*(1-BoEPredicted))
byTask = byTask %>% mutate(BoEErrorReduction = pmax(0,(BoEPredictedAccuracy-MajorityClass)/(1-MajorityClass)))


plot = ggplot(data=byTask, aes(y=BoEErrorReduction, x=MeanS3ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")

plot = ggplot(data=byTask, aes(y=BoEErrorReduction, x=CBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")



plot = ggplot(data=byTask, aes(y=BoEPredictedAccuracy, x=MeanS3ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")



plot = ggplot(data=byTask, aes(y=BoEPredicted, x=MeanS3ensitivity, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")


plot = ggplot(data=byTask, aes(x=BoEPredicted, y=100*IndivCBOWErrorReduction, color=Type)) + geom_point() + geom_label_repel(aes(label=Task), label.size=NA)
plot = plot + theme_bw()  + theme(legend.position="none")
plot = plot + ylab("BoE Error Reduction (%)") + xlab("Average Block Sensitivity")
#ggsave(plot, file="sensitivity-bowAccuracy.pdf", width=3, height=4)





#plot = ggplot(data=byTask, aes(x=MeanS3ensitivity, y=CBOWErrorReduction, color=Type)) + geom_point() + geom_label(aes(label=Task)) + theme_bw()


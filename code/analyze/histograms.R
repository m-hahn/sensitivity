

library(tidyr)
library(dplyr)
library(ggplot2)




cr = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="textclas")
mr = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="textclas")
subj = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="textclas")
mpqa = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="textclas")

cr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas_finetuned.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="textclas_finetuned")
mr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas_finetuned.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="textclas_finetuned")
subj_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas_finetuned.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="textclas_finetuned")
mpqa_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas_finetuned.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="textclas_finetuned")

data = rbind(cr, mr, mpqa, cr_finetuned, mr_finetuned, mpqa_finetuned , subj_finetuned, subj) 

plot = ggplot(data=data, aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type))
plot = ggplot(data=data %>% filter(FloatSensitivity != 0), aes(x=FloatSensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type))


cola = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_CoLA.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE", Tuned=FALSE)
cola2 = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_CoLA_Finetuned.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE", Tuned=TRUE)
data=rbind(cola, cola2)


plot = ggplot(data=data, aes(x=BinarySensitivity, group=Tuned, color=Tuned, linetype=Tuned)) + geom_density() + theme_bw()


mnli = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_MNLI_c.py", sep="\t") %>% mutate(Task = "MNLI", Type="GLUE")
qqp = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_QQP.py", sep="\t") %>% mutate(Task = "QQP", Type="GLUE")
rte = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_RTE_c.py", sep="\t", quote="#") %>% mutate(Task = "RTE", Type="GLUE")
wsc = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_evaluateRobertaSensitivity_New_c.py", sep="\t") %>% mutate(Task = "WSC", Type="GLUE")
cr = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="textclas")
mr = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="textclas")
subj = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="textclas")
mpqa = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_textclas.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="textclas")
cola = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_CoLA_Finetuned.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE")
mrpc = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_MRPC_c.py", sep="\t", quote="#") %>% mutate(Task = "MRPC", Type="GLUE")
sts = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_STS-B_c.py", sep="\t", quote="#") %>% mutate(Task = "STS-B", Type="GLUE")


parsing = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_parsing.py", sep="\t", quote="#") %>% mutate(Task = "Parsing_Rel", Type="parsing")
parsing_position = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_parsing_position.py", sep="\t", quote="#") %>% mutate(Task = "Parsing_Head", Type="parsing")


# performance of GPT2 is far from perfect on this task
gym248 = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_SyntaxGym_248_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym248", Type="Syntax")
gym260 = read.csv("~/CS_SCR/sensitivity/sensitivities/sensitivities_estimateSensitivity_SyntaxGym_260_gpt2.py", sep="\t", quote="#") %>% mutate(Task = "Gym260", Type="Syntax")

	

data = rbind(mnli, qqp, rte, wsc, cr_finetuned, subj_finetuned, mpqa_finetuned, cola, mrpc, sts, gym248, gym260, parsing, parsing_position, mr_finetuned)

library(stringr)
countWhitespace = function(string) {
	return(str_count(string, " "))
}


data$SentLength = Vectorize(countWhitespace)(as.character(data$Original))



plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinarySensitivity, group=Type, color=Type)) + geom_point() + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 100), aes(x=SentLength, y=BinarySensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw()
ggsave(plot, file="byLength_textclas_glue.pdf")

plot = ggplot(data=data %>% filter(SentLength < 100, Type == "textclas" | Task == "CoLA"), aes(x=SentLength, y=BinarySensitivity, group=Type, color=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)
plot = ggplot(data=data %>% filter(SentLength < 100, Type == "textclas" | Task == "CoLA"), aes(x=SentLength, y=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA) + theme_bw()
ggsave(plot, file="byLength_textclas_cola.pdf")
plot = ggplot(data=data %>% filter(SentLength < 100, Task == "CR" | Task == "CoLA"), aes(x=SentLength, y=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_smooth(method="loess",se=FALSE) + ylim(0,NA)


# a way to force monotonicity
# https://stats.stackexchange.com/questions/197509/how-to-smooth-data-and-force-monotonicity
#library(scam)
#SensitivityHat <- predict(scam(BinarySensitivity ~ s(SentLength, bs = "mpi"), data = data %>% filter(Task == "CR")))
#plot((data %>% filter(Task == "CR"))$SentLength, SensitivityHat)

data$Task = as.factor(data$Task)
data$Type = as.factor(data$Type)


plot = ggplot(data=data, aes(x=BinarySensitivity, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,10) + xlab("Block Sensitivity")
ggsave(plot, file="histogram_byTask.pdf")


plot = ggplot(data=data, aes(x=BinarySensitivity/SentLength, group=Task, color=Task)) + geom_density() + theme_bw() + facet_grid(rows=vars(Type)) + xlim(0,1) + xlab("Block Sensitivity / Length")



plot = ggplot(data=data %>% filter(Task=="Subj"), aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
plot = ggplot(data=data %>% filter(Task=="CR"), aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
plot = ggplot(data=data %>% filter(Task=="MPQA"), aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()
plot = ggplot(data=data %>% filter(Task=="Parsing"), aes(x=BinarySensitivity, group=Task, color=Task, linetype=Type)) + geom_density() + theme_bw()



byTask = data %>% group_by(Task, Type) %>% summarise(MeanSensitivity = mean(BinarySensitivity), MedianSensitivity = median(BinarySensitivity), SESensitivity = sd(BinarySensitivity)/sqrt(NROW(BinarySensitivity)))
byTask[order(byTask$MeanSensitivity),]

# TODO
# - STS-B
# - QNLI

# Tasks from Marvin & Linzen

# Is there a correlation between per-item sensitivity and performance of unigram-based model?

# Impact of base distribution / artifacts




bow = read.csv("bow.txt", sep="\t")

byTask = merge(byTask, bow, by=c("Task"), all=TRUE)

byTask = byTask %>% mutate(BOWErrorReduction = (Accuracy-MajorityClass)/(1-MajorityClass))

plot = ggplot(data=byTask, aes(x=MeanSensitivity, y=BOWErrorReduction, color=Type)) + geom_point() + geom_label(aes(label=Task)) + theme_bw()
ggsave(plot, file="sensitivity-bowAccuracy.pdf")





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

data = data %>% group_by(Task, Type, Original) %>% summarise(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE))


write.table(data,file="~/s1ens_models.R.tsv", sep="\t", quote=F)


library(stringr)
countWhitespace = function(string) {
	return(str_count(string, " "))
}


data$SentLength = Vectorize(countWhitespace)(as.character(data$Original))


library(lme4)
summary(lm(BinaryS1ensitivity ~ Type + SentLength + Task, data=data))

summary(lm(BinaryS1ensitivity ~ SentLength + Task, data=data %>% filter(Task %in% c("CoLA", "SST2"))))

summary(lm(BinaryS1ensitivity ~ SentLength + Type, data=data %>% filter(Type == "Text Cl." | Task == "CoLA")))


data$Task = as.factor(data$Task)
data$Type = as.factor(data$Type)



byTask = data %>% group_by(Task, Type) %>% summarise(MeanS1ensitivity = mean(BinaryS1ensitivity), MedianS1ensitivity = median(BinaryS1ensitivity), SES1ensitivity = sd(BinaryS1ensitivity)/sqrt(NROW(BinaryS1ensitivity)))
byTask[order(byTask$MeanS1ensitivity),]


write.table(byTask, file="xlnet-s1ensitivities.tsv", sep="\t", quote=F)



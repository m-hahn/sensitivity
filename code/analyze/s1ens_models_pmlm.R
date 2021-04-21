

library(tidyr)
library(dplyr)
library(ggplot2)


cr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas_PMLM_raw_1billion.py_cr", sep="\t", quote="#") %>% mutate(Task = "CR", Type="Text Cl.")
mr_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas_PMLM_raw_1billion.py_mr", sep="\t", quote="#") %>% mutate(Task = "MR", Type="Text Cl.")
subj_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas_PMLM_raw_1billion.py_subj", sep="\t", quote="#") %>% mutate(Task = "Subj", Type="Text Cl.")
mpqa_finetuned = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_textclas_PMLM_raw_1billion.py_mpqa", sep="\t", quote="#") %>% mutate(Task = "MPQA", Type="Text Cl.")

# GLUE
cola = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_CoLA_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "CoLA", Type="GLUE")
mnli = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_MNLI_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "MNLI", Type="GLUE")
qqp = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_QQP_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "QQP", Type="GLUE")
rte = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_RTE_getSensitivityParts_PMLM_raw_1billion_WithIndep.py", sep="\t", quote="#") %>% mutate(Task = "RTE", Type="GLUE")
wsc = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_WSC_getSensitivityParts_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "WSC", Type="GLUE")
#sts = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_STS-B_c.py", sep="\t", quote="#") %>% mutate(Task = "STS-B", Type="GLUE")
qnli = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_QNLI_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "QNLI", Type="GLUE")
sst2 = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion.py", sep="\t", quote="#") %>% mutate(Task = "SST2", Type="GLUE")
mrpc = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_MRPC_PMLM_raw_1billion_WithIndep.py", sep="\t", quote="#") %>% mutate(Task = "MRPC", Type="GLUE")

# Parsing
parsing = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_parsing_PMLM.py", sep="\t", quote="#") %>% mutate(Task = "Labels", Type="Parsing")
parsing_position = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_parsing_position_PMLM.py", sep="\t", quote="#") %>% mutate(Task = "Heads", Type="Parsing")
tagging = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_tagging_PMLM.py", sep="\t", quote="#") %>% mutate(Task = "Tagging", Type="Parsing")

# SyntaxGym
gym248 = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SyntaxGym_248_gpt2_PMLM.py", sep="\t", quote="#") %>% mutate(Task = "Gym248", Type="Syntax")
gym260 = read.csv("/home/user/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SyntaxGym_260_gpt2_PMLM.py", sep="\t", quote="#") %>% mutate(Task = "Gym260", Type="Syntax")

data = rbind(cr_finetuned, mr_finetuned, subj_finetuned, mpqa_finetuned, cola, mnli, qqp, rte, wsc, qnli, sst2, mrpc, parsing, parsing_position, tagging)
data = rbind(data, gym248, gym260)

data = data %>% group_by(Task, Type, Original) %>% summarise(BinaryS1ensitivity=mean(BinaryS1ensitivity, na.rm=TRUE))


write.table(data,file="~/s1ens_models_pmlm.R.tsv", sep="\t", quote=F)


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


write.table(byTask, file="pmlm-s1ensitivities.tsv", sep="\t", quote=F)



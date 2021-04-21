pmlm = read.csv("~/s1ens_models_pmlm.R.tsv", sep="\t", quote="#")
xlnet = read.csv("~/s1ens_models.R.tsv", sep="\t", quote="#")




library(stringr)
pmlm$Original = str_replace_all(pmlm$Original, "  ", " ")
pmlm$Original = str_replace_all(pmlm$Original, " </s>", "")
pmlm$Original = str_replace_all(pmlm$Original, "</s>", "")
xlnet$Original = str_replace_all(xlnet$Original, "</s>", "")


pmlm$Original = str_replace_all(pmlm$Original, " ", "")
xlnet$Original = str_replace_all(xlnet$Original, " ", "")


data = merge(pmlm, xlnet, by=c("Original", "Task", "Type"), all=TRUE)


data %>% group_by(Type, Task) %>% summarise(DataPoints = sum(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x)))

data = data %>% filter(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x))

library(tidyr)
library(dplyr)
library(ggplot2)


plot = ggplot(data, aes(x=BinaryS1ensitivity.x, y=BinaryS1ensitivity.y, color=Type, group=Type)) + geom_point(alpha=0.1) + geom_smooth(method="lm", se=F, aes(group=NULL, color=NULL))

data %>% group_by(Type, Task) %>% summarise(DataPoints = sum(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x)), Corr = cor(BinaryS1ensitivity.x, BinaryS1ensitivity.y, use="complete"))

data %>% group_by(Type) %>% summarise(DataPoints = sum(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x)), Corr = cor(BinaryS1ensitivity.x, BinaryS1ensitivity.y, use="complete"))

data %>% group_by() %>% summarise(DataPoints = sum(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x)), Corr = cor(BinaryS1ensitivity.x, BinaryS1ensitivity.y, use="complete"))

data %>% group_by(Type, Task) %>% summarise(DataPoints = sum(!is.na(BinaryS1ensitivity.y*BinaryS1ensitivity.x)), Corr = cor(BinaryS1ensitivity.x, BinaryS1ensitivity.y, use="complete")) %>% summarise(Corr=mean(Corr))


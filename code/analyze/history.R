


hist = read.csv("history.tsv", sep="\t")
tasks = read.csv("bow.txt", sep="\t")
hist = merge(hist, tasks, by=c("Task"), all.x=TRUE)
hist$Accuracy = hist$Accuracy/100
hist$Reduction = (hist$Accuracy-hist$MajorityClass) / (1-hist$MajorityClass)

xlnet_sens = read.csv("xlnet-s1ensitivities.tsv", sep="\t")
pmlm_sens = read.csv("pmlm_sensitivities.tsv", sep="\t")

sens = merge(xlnet_sens, pmlm_sens, by=c("Task"), all=TRUE) %>% mutate(Sensitivity = (MeanS1ensitivity+S1ensitivity)/2)

hist = merge(hist, sens, by=c("Task"), all=TRUE)
library(ggplot2)
library(tidyr)
library(dplyr)
ggplot(hist, aes(x=Year, y=Accuracy, group=Task)) + geom_line()
ggplot(hist, aes(x=Year, y=Reduction, group=Task, color=Sensitivity)) + geom_line() + theme_bw()

ggplot(hist, aes(x=Year, y=Reduction, group=Task, color=Type)) + geom_text(aes(label=Task)) + theme_bw() + ylab("Error Reduction")


ggplot(hist, aes(x=Year, y=Reduction, group=Task, color=Type)) + geom_line() + theme_bw() + ylab("Error Reduction")
ggsave("historical.pdf", height=3, width=4)
#ggplot(hist, aes(x=Year, y=Reduction, group=Task, color=Sensitivity)) + geom_line() + facet_wrap(~Sensitivity+Task)




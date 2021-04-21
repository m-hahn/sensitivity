library(tidyr)
library(dplyr)
library(ggplot2)


data7 = read.csv("../output/losses_learnability3_blockSens.py.tsv", sep="\t") %>% mutate(Sensitivity = AverageBlockSensitivity, Length=7)
data10 = read.csv("../output/losses_learnability3.py.tsv", sep="\t") %>% mutate(Sensitivity = AverageDegree, Length=10, AverageBlockSensitivity=NA)
data15 = read.csv("../output/losses_learnability3_2.py.tsv", sep="\t") %>% mutate(Sensitivity = AverageDegree, Length=15, AverageBlockSensitivity=NA)

data = rbind(data7, data10, data15)

# Number reported in the paper
cor(data7$AverageBlockSensitivity, data7$AverageDegree)

data = data %>% gather(Iterations, Loss, Acc100:Acc100000)


data$SensitivityCoarse = round(data$Sensitivity*2)/2
library(stringr)
data = data %>% mutate(Iterations = str_replace(Iterations, "Acc", ""))


plot = ggplot(data=data %>% group_by(SensitivityCoarse, Iterations, Length) %>% summarise(Loss = mean(Loss)), aes(x=SensitivityCoarse, y=Loss, group=Iterations, color=Iterations)) + geom_line() + theme_bw() + xlab("Sensitivity") + ylab("Mean Squared Error") +  theme(legend.position="bottom", legend.title=element_blank()) + facet_wrap(~Length, scales="free") #, ncol=1)
ggsave(plot, file="../output/learnability3_together.pdf", width=6, height=3)




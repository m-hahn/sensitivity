
#(base) user@user-X510UAR:~/Robustness-Low-Synergy-and-Cheap-Computation/code/learnability$ ls output/
#learnability3.pdf  losses_learnability3.py.tsv  randomFunct.tsv  randomInit.tsv
#(base) user@user-X510UAR:~/Robustness-Low-Synergy-and-Cheap-Computation/code/learnability$ 

library(dplyr)
library(tidyr)
dataLSTM = read.csv("output/randomInit.tsv", sep="\t") %>% mutate(Type = "LSTM")
dataUniform = read.csv("output/randomFunct.tsv", sep="\t") %>% mutate(Type = "Uniform")


data= rbind(dataLSTM, dataUniform)


library(ggplot2)

plot = ggplot(data, aes(x=AverageBlockSensitivity, fill=NULL, color=Type))
plot = plot + theme_classic()
plot = plot + xlab("Average Block Sensitivity") + ylab("Density") + xlim(0,7)
plot = plot + theme(legend.position="bottom")
plot = plot + geom_density(data= data, aes(y=..scaled..))
ggsave("output/lstm-init.pdf", width=4, height=2.5)


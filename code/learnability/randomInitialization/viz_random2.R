
#(base) user@user-X510UAR:~/Robustness-Low-Synergy-and-Cheap-Computation/code/learnability$ ls output/
#learnability3.pdf  losses_learnability3.py.tsv  randomFunct.tsv  randomInit.tsv
#(base) user@user-X510UAR:~/Robustness-Low-Synergy-and-Cheap-Computation/code/learnability$ 

library(dplyr)
library(tidyr)
dataLSTM1 = read.csv("output/randomInit6_S3ensitivity_Median.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Uniform)", Dimensions="128")
dataLSTM2 = read.csv("output/randomInit6_S3ensitivity_Median_2.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Normal)", Dimensions="128")
#dataLSTM3 = read.csv("output/randomInit6_S3ensitivity_Median_3.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Bernoulli)", Dimensions="128")
dataLSTM1d = read.csv("output/randomInit6_S3ensitivity_Median_Dim.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Uniform)", Dimensions="256")
dataLSTM2d = read.csv("output/randomInit6_S3ensitivity_Median_2_Dim.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Normal)", Dimensions="256")
#dataLSTM3d = read.csv("output/randomInit6_S3ensitivity_Median_3_Dim.py.tsv", sep="\t") %>% mutate(Type = "LSTM", Initialization="LSTM (Bernoulli)", Dimensions="256")
dataUniform = read.csv("output/randomFunct_S3ensitivity.tsv", sep="\t") %>% mutate(Type = "Uniform", Initialization="Uniform", Dimensions="0")


data= rbind(dataLSTM1, dataLSTM2, dataLSTM1d, dataLSTM2d, dataUniform)


library(ggplot2)

plot = ggplot(data, aes(x=AverageBlockS3ensitivity, linetype=Initialization, colour=Dimensions))
plot = plot + theme_classic()
plot = plot + xlab("Average Block Sensitivity") + ylab("Density") + xlim(1,7)
#plot = plot + theme(legend.position="bottom")
plot = plot + geom_density(data= data, aes(y=..scaled..))
ggsave("output/lstm-init.pdf", width=5, height=2.5)


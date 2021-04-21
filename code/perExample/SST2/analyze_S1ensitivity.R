data = read.csv("../outputs/perExSST2_S1ensitivity.py.tsv", sep="\t")
library(tidyr)
library(dplyr)
data = data %>% mutate(SubspansPosFrac = SubspansPos/(SubspansPos+SubspansNeg))
data = data %>% mutate(SubspansDev = 2*sqrt(SubspansPosFrac*(1-SubspansPosFrac)))

sink("output/analyze_S1ensitivity.R_sens_dispersion.txt")
print(summary(lm(FloatS1ensitivity ~ SubspansDev + Length, data=data)))
sink()

data = data %>% mutate(BOWCorrect = (BOWPrediction == Label))
data = data %>% mutate(RobertaCorrect = (RobertaPrediction == Label))

sink("output/analyze_S1ensitivity.R_correct_s1ensitivity.txt")
print(summary(glm(RobertaCorrect ~ FloatS1ensitivity + Length, family="binomial", data=data)))
print(summary(glm(BOWCorrect ~ FloatS1ensitivity + Length, family="binomial", data=data)))
sink()

data2 = rbind(data %>% rename(Correct = RobertaCorrect) %>% mutate(Model="RoBERTa", BOWCorrect=NULL), data %>% rename(Correct=BOWCorrect) %>% mutate(Model="CBOW", RobertaCorrect=NULL)) %>% mutate(CorrectF = Correct+0)

data2 = data2 %>% mutate(ModelC = ifelse(Model == "CBOW", 0.5, -0.5), CorrectF=Correct+0.0)
library(lme4)
#library(brms)
#summary(brm(CorrectF ~ FloatS1ensitivity*ModelC + Length*ModelC + (1+FloatS1ensitivity+ModelC+FloatS1ensitivity*ModelC + Length*ModelC|Sentence), family="bernoulli", data=data2))


library(ggplot2)
data = data %>% mutate(BOWCorrectF = BOWCorrect+0)
plot = ggplot(data, aes(y=BOWCorrectF, x=FloatS1ensitivity)) + geom_smooth()


plot = ggplot(data2, aes(y=CorrectF, x=FloatS1ensitivity, group=Model, color=Model)) +
     geom_smooth(    method = "glm", method.args = list(family = "binomial")) + theme_bw() + ylab("Accuracy") + xlab("Sensitivity")
ggsave(plot, file="../outputs/s1ensitivity_accuracy_roberta-cbow.pdf", width=3, height=2)

# Sanity checking the smoothed graph. Excluding bin 5 because there is only one datapoint there
data3 = data2 %>% mutate(Bin = round(FloatS1ensitivity)) %>% group_by(Model, Bin) %>% summarise(CorrectF = mean(CorrectF)) %>% mutate(FloatS1ensitivity = Bin) %>% filter(Bin<5)

plot = ggplot(data3, aes(y=CorrectF, x=FloatS1ensitivity, group=Model, color=Model)) +
     geom_line() + theme_bw() + ylab("Accuracy") + xlab("Sensitivity")



#plot = ggplot(data, aes(y=BOWCorrectF, x=FloatS1ensitivity)) +
#     geom_smooth(    method = "glm", method.args = list(family = "binomial")) + theme_bw() + ylab("CBOW Accuracy") + xlab("Sensitivity")
#ggsave(plot, file="../outputs/s1ensitivity_accuracy.pdf", width=2, height=2)
#
#plot = ggplot(data, aes(x=SubspansDev, y=FloatS1ensitivity)) + geom_smooth() + theme_bw() + xlab("Dispersion") + ylab("Sensitivity")
#ggsave(plot, file="../outputs/subspans_s1ensitivity.pdf", width=2, height=2)

#plot = ggplot(data, aes(y=SubspansDev, x=FloatS1ensitivity)) + geom_smooth() + theme_bw() + ylab("Dispersion") + xlab("Sensitivity") + xlim(0,5)
#ggsave(plot, file="../outputs/subspans_s1ensitivity_rev.pdf", width=2, height=2)

plot = ggplot(data, aes(y=SubspansDev, x=FloatS1ensitivity)) + geom_point(color="gray") + geom_smooth(se=F) + theme_bw() + ylab("Dispersion") + xlab("Sensitivity") + xlim(NA,3) + ylim(0,1)
ggsave(plot, file="../outputs/subspans_s1ensitivity_rev.pdf", width=2, height=2)


#u = data %>% filter(SubspansDev > 0)

#plot = ggplot(u, aes(y=SubspansDev, x=FloatS1ensitivity)) + geom_smooth() + theme_bw() + ylab("Dispersion") + xlab("Sensitivity") + xlim(0,5)

#print(summary(lm(FloatS1ensitivity ~ SubspansDev + Length, data=u)))


#plot = ggplot(data, aes(x=SubspansDev, y=FloatS1ensitivity)) + geom_smooth() + geom_point() + theme_bw() + xlab("Dispersion") + ylab("Sensitivity")


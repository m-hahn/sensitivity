data = read.csv("outputs/perExSST2.py.tsv", sep="\t")
library(tidyr)
library(dplyr)
data = data %>% mutate(SubspansPosFrac = SubspansPos/(SubspansPos+SubspansNeg))
data = data %>% mutate(SubspansDev = 2*sqrt(SubspansPosFrac*(1-SubspansPosFrac)))


summary(lm(FloatSensitivity ~ SubspansDev + Length, data=data))

data = data %>% mutate(BOWCorrect = (BOWPrediction == Label))
data = data %>% mutate(RobertaCorrect = (RobertaPrediction == Label))

summary(glm(BOWCorrect ~ SubspansDev + FloatSensitivity + Length, family="binomial", data=data))
summary(glm(RobertaCorrect ~ SubspansDev + FloatSensitivity + Length, family="binomial", data=data))

summary(glm(RobertaCorrect ~ FloatSensitivity + Length, family="binomial", data=data))
summary(glm(BOWCorrect ~ FloatSensitivity + Length, family="binomial", data=data))

data2 = rbind(data %>% rename(Correct = RobertaCorrect) %>% mutate(Model="RoBERTa", BOWCorrect=NULL), data %>% rename(Correct=BOWCorrect) %>% mutate(Model="CBOW", RobertaCorrect=NULL)) %>% mutate(CorrectF = Correct+0)

data2 = data2 %>% mutate(ModelC = ifelse(Model == "CBOW", 0.5, -0.5), CorrectF=Correct+0.0)
library(lme4)
#library(brms)
#summary(brm(CorrectF ~ FloatSensitivity*ModelC + Length*ModelC + (1+FloatSensitivity+ModelC+FloatSensitivity*ModelC + Length*ModelC|Sentence), family="bernoulli", data=data2))


library(ggplot2)
data = data %>% mutate(BOWCorrectF = BOWCorrect+0)
plot = ggplot(data, aes(y=BOWCorrectF, x=FloatSensitivity)) + geom_smooth()


plot = ggplot(data2, aes(y=CorrectF, x=FloatSensitivity, group=Model, color=Model)) +
     geom_smooth(    method = "glm", method.args = list(family = "binomial")) + theme_bw() + ylab("Accuracy") + xlab("Sensitivity")
ggsave(plot, file="outputs/sensitivity_accuracy_roberta-cbow.pdf", width=3, height=2)

# Sanity checking the smoothed graph. Excluding bin 5 because there is only one datapoint there
data3 = data2 %>% mutate(Bin = round(FloatSensitivity)) %>% group_by(Model, Bin) %>% summarise(CorrectF = mean(CorrectF)) %>% mutate(FloatSensitivity = Bin) %>% filter(Bin<5)

plot = ggplot(data3, aes(y=CorrectF, x=FloatSensitivity, group=Model, color=Model)) +
     geom_line() + theme_bw() + ylab("Accuracy") + xlab("Sensitivity")



plot = ggplot(data, aes(y=BOWCorrectF, x=FloatSensitivity)) +
     geom_smooth(    method = "glm", method.args = list(family = "binomial")) + theme_bw() + ylab("CBOW Accuracy") + xlab("Sensitivity")
ggsave(plot, file="outputs/sensitivity_accuracy.pdf", width=2, height=2)

plot = ggplot(data, aes(x=BOWCorrect, y=FloatSensitivity)) + geom_violin()

plot = ggplot(data, aes(x=BOWCorrect, y=SubspansDev)) + geom_violin()

plot = ggplot(data, aes(x=SubspansDev, y=FloatSensitivity)) + geom_smooth() + theme_bw() + xlab("Dispersion") + ylab("Sensitivity")
ggsave(plot, file="outputs/subspans_sensitivity.pdf", width=2, height=2)

plot = ggplot(data, aes(y=SubspansDev, x=FloatSensitivity)) + geom_smooth() + theme_bw() + ylab("Dispersion") + xlab("Sensitivity") #+ ylim(0,1)
ggsave(plot, file="outputs/subspans_sensitivity_rev.pdf", width=2, height=2)


plot = ggplot(data, aes(x=Length, y=FloatSensitivity)) + geom_smooth() + theme_bw()





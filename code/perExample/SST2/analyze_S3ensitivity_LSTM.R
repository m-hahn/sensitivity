data = read.csv("../outputs/perExSST2_S3ensitivity.py.tsv", sep="\t")
dataLSTM = read.csv("../outputs/perExSST2_S3ensitivity_LSTM.py.tsv", sep="\t")
library(tidyr)
library(dplyr)
data= merge(data, dataLSTM, by=c("Sentence", "Label"), all.x=TRUE)
data = data %>% mutate(SubspansPosFrac = SubspansPos/(SubspansPos+SubspansNeg))

data = data %>% mutate(SubspansDev = 2*sqrt(SubspansPosFrac*(1-SubspansPosFrac)))


summary(lm(FloatS3ensitivity ~ SubspansDev + Length, data=data))

data = data %>% mutate(LSTMCorrect = (LSTMPrediction == Label))
data = data %>% mutate(BOWCorrect = (BOWPrediction == Label))
data = data %>% mutate(RobertaCorrect = (RobertaPrediction == Label))

summary(glm(RobertaCorrect ~ FloatS3ensitivity + Length, family="binomial", data=data))
summary(glm(BOWCorrect ~ FloatS3ensitivity + Length, family="binomial", data=data))
summary(glm(LSTMCorrect ~ FloatS3ensitivity + Length, family="binomial", data=data))

data2 = rbind(data %>% rename(Correct = RobertaCorrect) %>% mutate(Model="RoBERTa", BOWCorrect=NULL, LSTMCorrect=NULL), data %>% rename(Correct=BOWCorrect) %>% mutate(Model="BoE", RobertaCorrect=NULL, LSTMCorrect=NULL)) 
data2 = rbind(data2, data %>% rename(Correct = LSTMCorrect) %>% mutate(Model="LSTM", BOWCorrect=NULL, RobertaCorrect=NULL))%>% mutate(CorrectF = Correct+0)

data2 = data2 %>% mutate(ModelC = ifelse(Model == "CBOW", 0.5, -0.5), CorrectF=Correct+0.0)
library(lme4)
#library(brms)
#summary(brm(CorrectF ~ FloatS3ensitivity*ModelC + Length*ModelC + (1+FloatS3ensitivity+ModelC+FloatS3ensitivity*ModelC + Length*ModelC|Sentence), family="bernoulli", data=data2))


library(ggplot2)
data = data %>% mutate(BOWCorrectF = BOWCorrect+0)
plot = ggplot(data, aes(y=BOWCorrectF, x=FloatS3ensitivity)) + geom_smooth()


plot = ggplot(data2, aes(y=CorrectF, x=FloatS3ensitivity, group=Model, color=Model)) +
     geom_smooth(    method = "glm", method.args = list(family = "binomial")) + theme_bw() + ylab("Accuracy") + xlab("Sensitivity")
ggsave(plot, file="../outputs/s3ensitivity_accuracy_roberta-cbow-lstm.pdf", width=3, height=2)


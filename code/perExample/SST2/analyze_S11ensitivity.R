data = read.csv("../outputs/perExSST2_S11ensitivity.py.tsv", sep="\t")
data3 = read.csv("../outputs/perExSST2_S1ensitivity.py.tsv", sep="\t")
data = data %>% mutate(SubspansPosFrac = SubspansPos/(SubspansPos+SubspansNeg))
data = data %>% mutate(SubspansDev = 2*sqrt(SubspansPosFrac*(1-SubspansPosFrac)))


data = merge(data, data3, by=c("Sentence", "Length", "Label", "FloatSensitivity"))
library(tidyr)
library(dplyr)


summary(lm(FloatS11ensitivity ~ SubspansDev + Length, data=data))

data = data %>% mutate(BOWCorrect = (BOWPrediction.x == Label))
data = data %>% mutate(RobertaCorrect = (RobertaPrediction.x == Label))


sink("output/analyze_S11ensitivity.R_correct_s1ensitivity.txt")
model1 = (glm(BOWCorrect ~ FloatS11ensitivity + Length, family="binomial", data=data))                         
model2 = (glm(BOWCorrect ~ FloatS1ensitivity + Length, family="binomial", data=data))                         
print(summary(model1))
print(BIC(model2)-BIC(model1))
sink()

summary(glm(RobertaCorrect ~ FloatS11ensitivity + Length, family="binomial", data=data))
summary(glm(BOWCorrect ~ FloatS11ensitivity + Length, family="binomial", data=data))



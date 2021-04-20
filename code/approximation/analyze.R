library(dplyr)
library(dplyr)
library(ggplot2)
d7 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant7.py", sep="\t", quote="#") %>% mutate(Group=7)
d6 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant6.py", sep="\t", quote="#") %>% mutate(Group=6)
d5 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant5.py", sep="\t", quote="#") %>% mutate(Group=5)
d4 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant4.py", sep="\t", quote="#") %>% mutate(Group=4)
d3 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant3.py", sep="\t", quote="#") %>% mutate(Group=3)
d2 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant2.py", sep="\t", quote="#") %>% mutate(Group=2)
d1 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed_Variant1.py", sep="\t", quote="#") %>% mutate(Group=1)
d0 = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811_Allowed.py", sep="\t", quote="#") %>% mutate(Group=0)
dInf = read.csv("~/CS_SCR/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_PMLM_raw_1billion_811.py", sep="\t", quote="#") %>% mutate(Group=-1)


data = rbind(d7, d6, d5, d4, d3, d2, d1, d0, dInf)

mean(dInf$BinaryS1ensitivity)
mean(d0$BinaryS1ensitivity)
mean(d7$BinaryS1ensitivity)


#> cor.test(dInf$BinaryS1ensitivity, d0$BinaryS1ensitivity)
#
#	Pearson's product-moment correlation
#
#data:  dInf$BinaryS1ensitivity and d0$BinaryS1ensitivity
#t = 13.322, df = 18, p-value = 9.224e-11
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# 0.8823830 0.9815085
#sample estimates:
#      cor
#0.9528441


cor.test(dInf$BinaryS1ensitivity, d7$BinaryS1ensitivity)


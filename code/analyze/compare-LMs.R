library(dplyr)
library(tidyr)
library(ggplot2)


xlnet = read.csv("xlnet-s1ensitivities.tsv", sep="\t")%>% rename(XLNet_MeanS1ensitivity=MeanS1ensitivity)
pmlm = read.csv("pmlm_sensitivities.tsv", sep="\t") %>% rename(PMLM_MeanS1ensitivity=S1ensitivity)

data = merge(xlnet, pmlm, by=c("Task"))
library(ggrepel)

plot = ggplot(data, aes(x=XLNet_MeanS1ensitivity, y = PMLM_MeanS1ensitivity, color=Type, group=Type)) + geom_point() + geom_text_repel(aes(label=Task)) + theme_bw() + theme( legend.position="none") + xlab("Estimated using XLNet") + ylab("Estimated using u-PMLM")
ggsave(plot, file="comparing-xlnet-pmlm", weight=3, height=3)

cor.test(data$XLNet_MeanS1ensitivity, data$PMLM_MeanS1ensitivity)



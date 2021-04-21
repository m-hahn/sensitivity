lstm = read.csv("/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_lstm_getSensitivityParts_finetuned.py", sep="\t", quote="@") %>% rename(LSTM_Sensitivity = BinaryS1ensitivity)
roberta = read.csv("/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_estimateS1ensitivity_SST2_getSensitivityParts_finetuned_New.py", sep="\t", quote="@")

library(dplyr)
library(tidyr)
library(ggplot2)

# estimateS1ensitivity_RTE_cd_LSTM_getSensitivityParts.py
# (base) mhahn@sc:/juice/scr/mhahn/CODE/fairseq$ vimdiff estimateS1ensitivity_RTE_cd_LSTM_getSensitivityParts.py estimateS1ensitivity_RTE_cd.py

data = merge(lstm, roberta, by=c("Original"), all=T) %>% filter(BinaryS1ensitivity > 0.2) # Outlier

cor.test(data$LSTM_Sensitivity, data$BinaryS1ensitivity)

plot = ggplot(data, aes(x=BinaryS1ensitivity, y=LSTM_Sensitivity))+  geom_point(alpha=0.3)+ geom_smooth()  + theme_bw() + xlab("RoBERTa Sensitivity") + ylab("BiLSTM Sensitivity")
ggsave(plot, file="roberta-lstm-sst2.pdf", height=3, width=3)

write.table(data, file="lstm-sst2.tsv", sep="\t", quote=F)

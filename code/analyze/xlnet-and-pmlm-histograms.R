

library(tidyr)
library(dplyr)
library(ggplot2)


library(stringr)
countWhitespace = function(string) {
	return(str_count(string, " "))
}


pmlm = read.csv("~/s1ens_models_pmlm.R.tsv", sep="\t", quote="#") %>% mutate(Model = "u-PMLM")
xlnet = read.csv("~/s1ens_models.R.tsv", sep="\t", quote="#") %>% mutate(Model = "XLNet")

data = rbind(pmlm, xlnet)

data$Type = as.character(data$Type)
data[data$Type == "Text Cl.",]$Type = "Textcl"

data$Task = as.factor(data$Task)
data$Type = as.factor(data$Type)





library(ggpubr)

library(cowplot)


for(model in c("u-PMLM", "XLNet")) {
  for(type in c("GLUE", "Parsing", "Syntax", "Textcl")) {
   plot = ggplot(data=data %>% filter(Model == model, Type == type), aes(x=BinaryS1ensitivity, group=Task, color=Task)) + geom_density(aes(y=..scaled..)) + theme_bw() + xlim(0,6.5) + xlab(NULL) + theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())
   legend <- get_legend(plot+ theme(legend.title = element_blank ())  + guides(color=guide_legend(nrow=1,byrow=TRUE)))
   legend = plot_grid(NULL, legend)
   ggsave(legend, file=paste("joint_", type, "_legend.pdf", sep=""), width=4, height=0.5)

   plot = plot + theme(legend.position="none", text=element_text(size=16))


   if(type == "Text Cl.") {
	   type = "textclas"
   }
   ggsave(plot, file=paste("joint_", type, "_", model, ".pdf", sep=""), height=1.5, width=3.5)
  }
}


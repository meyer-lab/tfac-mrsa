# Gene Ontology Analysis of Transcriptional Components
library(clusterProfiler)
library(ggplot2)

data <- read.csv("tfac/data/mrsa/MRSA_GO_Input.csv")
data <- data[, c(1, 2)]
xx <- compareCluster(Gene~Components, data=data, fun='enrichGO', keyType="ENSEMBL", ont="BP",  OrgDb='org.Hs.eg.db', pAdjustMethod='hochberg')
xxSimp <- simplify(xx, cutoff=0.7, by="p.adjust", select_fun=min)

ggplot(xxSimp, aes(x=Components, y=Description, colour=p.adjust)) + geom_point(aes(size=GeneRatio)) + theme_bw() + ggtitle("Biological Process") + theme(plot.title = element_text(hjust = 0.5))

OutputPath = '/Users/foo/Desktop/MRSA_GO_BP.png'
ggsave(OutputPath, units='cm', width=20, height=15)

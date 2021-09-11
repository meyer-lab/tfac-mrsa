# Gene Ontology Analysis of Transcriptional Components
library(clusterProfiler)
library(ggplot2)

data <- read.csv("tfac/data/mrsa/MRSA_gsea_input_ENTREZ.csv")
data <- data[, c(1, 2)]
xx <- compareCluster(entrezgene~Components, data=data, fun="enrichWP", pAdjustMethod='bonferroni', organism="Homo sapiens")

ggplot(xx, aes(x=Components, y=Description, colour=p.adjust)) + geom_point(aes(size=GeneRatio)) + theme_bw() + ggtitle("Biological Process") + theme(plot.title = element_text(hjust = 0.5))

ggsave("MRSA_gsea_WP.png", units='cm', width=20, height=15)

"""If entering custom gene signature set: """
# sigdb <- read.gmt("tfac/data/mrsa/c7.immunesigdb.v7.4.entrez.gmt")
# xx <- compareCluster(EntrezID~Components, data=data, fun=enricher, TERM2GENE=sigdb, pAdjustMethod='bonferroni', minGSSize=100, maxGSSize=500)
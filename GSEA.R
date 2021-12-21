# Gene Ontology Analysis of Transcriptional Components
library(clusterProfiler)
library(ggplot2)

gsp <- read.gmt("tfac/data/gsea_libraries/gene_set_paper.gmt")
azi <- read.gmt("tfac/data/gsea_libraries/gsea_libraries/Azimuth_Cell_Types_2021_short.gmt")
peri <- read.gmt("tfac/data/gsea_libraries/gsea_libraries/CellMarker_Augmented_2021_peripheral.gmt")
hub <- read.gmt("tfac/data/gsea_libraries/gsea_libraries/HuBMAP_ASCT_plus_B_augmented_w_RNAseq_Coexpression_immune_bloodonly.gmt")
bd <- read.gmt("tfac/data/gsea_libraries/gsea_libraries/CellMarker_Augmented_2021_blood.gmt")

data <- read.csv("tfac/data/mrsa/MRSA_gsea_input_ENTREZ.csv")
# xx <- compareCluster(EntrezID~Components, data=data, fun="enrichWP", pAdjustMethod='bonferroni', organism="Homo sapiens")

xx <- compareCluster(entrezgene~Components, data=data, fun=enricher, TERM2GENE=bd, pAdjustMethod='hochberg')

ggplot(xx, aes(x=Components, y=Description, colour=p.adjust)) + geom_point(aes(size=GeneRatio)) + theme_bw() + ggtitle("Biological Process") + theme(plot.title = element_text(hjust = 0.5))
OutputPath = 'foo'
ggsave(OutputPath, units='cm', width=20, height=15)

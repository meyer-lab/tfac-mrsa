---
title: Multi-omic patterns improve prediction of MRSA infection outcome
keywords:
- MRSA
lang: en-US
date-meta: '2021-05-04'
author-meta:
- Scott D. Taylor
- Elaine F. Reed
- Aaron S. Meyer
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Multi-omic patterns improve prediction of MRSA infection outcome" />
  <meta name="citation_title" content="Multi-omic patterns improve prediction of MRSA infection outcome" />
  <meta property="og:title" content="Multi-omic patterns improve prediction of MRSA infection outcome" />
  <meta property="twitter:title" content="Multi-omic patterns improve prediction of MRSA infection outcome" />
  <meta name="dc.date" content="2021-05-04" />
  <meta name="citation_publication_date" content="2021-05-04" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Scott D. Taylor" />
  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />
  <meta name="citation_author" content="Elaine F. Reed" />
  <meta name="citation_author_institution" content="UCLA Immunogenetics Center, Department of Pathology and Laboratory Medicine, University of California Los Angeles, Los Angeles, CA, USA" />
  <meta name="citation_author" content="Aaron S. Meyer" />
  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Department of Bioinformatics, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Jonsson Comprehensive Cancer Center, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles" />
  <meta name="citation_author_orcid" content="0000-0003-4513-1840" />
  <meta name="twitter:creator" content="@aarmey" />
  <link rel="canonical" href="https://meyer-lab.github.io/tfac-mrsa/" />
  <meta property="og:url" content="https://meyer-lab.github.io/tfac-mrsa/" />
  <meta property="twitter:url" content="https://meyer-lab.github.io/tfac-mrsa/" />
  <meta name="citation_fulltext_html_url" content="https://meyer-lab.github.io/tfac-mrsa/" />
  <meta name="citation_pdf_url" content="https://meyer-lab.github.io/tfac-mrsa/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://meyer-lab.github.io/tfac-mrsa/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://meyer-lab.github.io/tfac-mrsa/v/d8330cb4446098641897e108adeb9feb66bba1d3/" />
  <meta name="manubot_html_url_versioned" content="https://meyer-lab.github.io/tfac-mrsa/v/d8330cb4446098641897e108adeb9feb66bba1d3/" />
  <meta name="manubot_pdf_url_versioned" content="https://meyer-lab.github.io/tfac-mrsa/v/d8330cb4446098641897e108adeb9feb66bba1d3/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography: []
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://meyer-lab.github.io/tfac-mrsa/v/d8330cb4446098641897e108adeb9feb66bba1d3/))
was automatically generated
from [meyer-lab/tfac-mrsa@d8330cb](https://github.com/meyer-lab/tfac-mrsa/tree/d8330cb4446098641897e108adeb9feb66bba1d3)
on May 4, 2021.
</em></small>

## Authors



+ **Scott D. Taylor**<br>
    · Github
    [scottdtaylor95](https://github.com/scottdtaylor95)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles
  </small>

+ **Elaine F. Reed**<br><br>
  <small>
     UCLA Immunogenetics Center, Department of Pathology and Laboratory Medicine, University of California Los Angeles, Los Angeles, CA, USA
  </small>

+ **Aaron S. Meyer**<br>
    ORCID
    [0000-0003-4513-1840](https://orcid.org/0000-0003-4513-1840)
    · Github
    [aarmey](https://github.com/aarmey)
    · twitter
    [aarmey](https://twitter.com/aarmey)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles; Department of Bioinformatics, University of California, Los Angeles; Jonsson Comprehensive Cancer Center, University of California, Los Angeles; Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles
  </small>



## Abstract {.page_break_before}




## Introduction

<!-- General introduction. -->

Methicillin-resistant Staphylococcus aureus (MRSA) bacteremia is a common, life-threatening infection, arising through both community-acquired and healthcare-associated settings [@PMID:23663462; @PMID:22760291]. These infections are associated with poor outcomes, and 15–30% of patients fail treatment with anti-MRSA antibiotics despite these drugs' effectiveness _in vitro_ [@DOI:10.1086/423145]. Infections that resolve under anti-MRSA treatment can be designated antibiotic-resolving (ARMB) or antibiotic-persistent (APMB) MRSA bacteremia [CITE]. The limited predictive value of in vitro susceptibility indicates a need to better understand the determinants of infection course _in vivo_.

<!-- What is known about disease course. -->

Recent progress has been made in identifying determinants of MRSA infection outcome. Host factors appear to play an important role, as patient outcome is independent of strain susceptibility to vancomycin or daptomycin treatment [@DOI:10.1086/423145]. Several demographic and clinical factors are associated with outcome but are insufficiently predictive to determine outcome in individual patients [CITE]. Based upon the hypothesis that outcome is determined by the confluence of host, pathogen, and antibiotic features, we have previously undertaken broad molecular profiling to integrate the molecular differences of infection response [CITE]. Our systems-level analysis has previously identified genetic [@DOI:10.1073/pnas.1909849116], transcriptional [CITE], and cytokine [CITE Rossetti] predictors of SAB persistence. However, the extent to which these signatures operate as distinct molecular mechanisms of immune phenotypic response, or reflect a shared underlying molecular mechanism, is yet unclear. Furthermore, we surmised that patterns across these molecular features might refine improve predictions of outcome if they helped to refine shared molecular mechanisms.

<!-- Data fusion / tensor models -->

Factorization-based models offer a natural solution to identifying patterns across datasets and data types. Most generally, these methods identify a reduced set of matrices that, when recombined, approximate the original measurements. Doing so, when appropriately matched to the structure of the data, helps to visualize its variation, reduce noise, impute missing values, and reduce dimensionality. Principal components analysis and non-negative matrix factorization are just two examples widely applied with data in matrix form. When integrating data across additional dimensions, higher-order generalizations of these methods exist [@doi:10.1137/07070111X]. These types of models are naturally suited to combining different types of data measurements, as each data type can be aligned along a dimension representing different samples. For example, PARAFAC2 models are particularly effective in combining data of different types as they allow each dataset to have a shared dimension such as patient subjects but then vary in the number of measurements within each dataset [CITE].

<!-- Introduction to the paper. -->

Here, we use a tensor factorization approach to integrate the transcriptional and cytokine responses to MRSA infection. Data integration allows us to identify consistent patterns of immunologic response across both types of data. Specifically, we identify a distinct innate and adaptive immune signature, present in both types of data, that together predict disease outcome. The combined data model predicts outcome better than either data type on its own, and its predictive accuracy is validated in an independent cohort. In total, this provides an approach to identifying integrative signatures of clinical immune response and dysfunction.


## Results

### A strategy for incorporating heterogeneous clinical measurements

Text.

![**Structure data decomposition can integrate clinical measurements with varying degrees of overlap.** A) General description of the data. Subjects with a MRSA infection had samples collected at admission, and then were monitored during their disease course. Identified subjects in the study were XXXX matched. Samples from each subject included serum, plasma, and isolated PBMCs. Inidividual subjects had cytokine measurements from either serum, plasma, or both. Similarly, PBMC gene expression was available for many but not all subjects. These measurements were then used to predict disease outcome, defined as the infection resolving (ARMB) or persisting (APMB) within XXX months. B) Overall structure of the data. Cytokine measurements from either plasma or serum can be arranged in a three-dimensional tensor, wherein one dimension indicates subject, cytokine, or sample source, each. In parallel, gene expression measurements are aligned with cytokine measurements along the subject dimension but not across genes. C) Data reduction is performed by identifying additively-separable components represented by the outer product of vectors along each dimension. The subjects dimension is shared across both the tensor and matrix reconstruction. Venn diagram of the variance explained by each factorization method. Canonical polyadic (CP) decomposition can explain the variation present within either the antigen-specific tensor or glycan matrix on their own [@PMID:18003902]. CMTF allows one to explain the shared variation between the matrix and tensor [@PMID:31251750]. In contrast, here we wish to explain the total variation across both the tensor and matrix. This is accomplished with TMTF [@DOI:10.1101/2021.01.03.425138].](figure1.svg "Figure 1"){#fig:figure1 width="100%"}

### TMTF factorization recovers consistent patterns across data types

Text.

![**TMTF factorization recovers consistent patterns across data types.** A) Number of components used in the TMTF decomposition versus the percent variance reconstructed (R2X). B–D) Decomposed components along the subjects (B), cytokines (C), and sample source (D) dimensions. Subjects are labeled according to the outcome of their infection and the cohort from which they were derived.](figure2.svg "Figure 2"){#fig:figure2 width="100%"}

### TMTF-based data integration improves predictions of patient outcomes

Text.

![**TMTF-based data integration improves predictions of patient outcomes.** A) Prediction accuracy when using the cytokines or gene expression measurements on their own, or when combined using TMTF. Accuracy was assessed by cross-validation within the training cohort. B) Prediction ROC curve based on cross-validation within the training cohort. C) ROC curve from validation cohort. C) Prediction accuracy using all pairs of components from TMTF. D) The prediction decision function with respect optimal pair of components for prediction.](figure3.svg "Figure 3"){#fig:figure3 width="100%"}

### MRSA infection response involves combined immune cell-cytokine feedback

Text.

![**A combined immune cell-cytokine regulatory program.** AAA. BBB. CCC.](figure4.svg "Figure 4"){#fig:figure4 width="100%"}


## Discussion




## Methods

### Patients and sample collection

This case-controlled study consisted of 58 SAB patients (28 APMB and 30 ARMB) propensity matched by sex, race, age, haemodialysis status, Type I diabetes, presence of an implantable device.  Details of clinical characteristics of study cohort are presented in Table 1. SAB cases were evaluated and consented for enrolment in the S. aureus Bacteremia Group (SABG) biorepository at Duke University Medical Centre (DUMC). Cases for the current study were carefully selected based on the following inclusion criteria: laboratory confirmed MRSA bacteremia; received appropriate vancomycin therapy; enrolled in the SABG study between 2007 and 2017 (to ensure contemporary medical practices). APMB was defined as patients had continuous MRSA positive blood cultures for at least 5 days after vancomycin antibiotic treatment [@DOI:10.1073/pnas.1909849116]; while ARMB patients had initial blood cultures that were positive for MRSA, but the subsequent blood cultures were negative.

### Molecular analysis

#### Luminex-based cytokine measurement

Human 38-plex magnetic cytokine/chemokine kits (EMD Millipore, HCYTMAG-60K- PX38) were used per manufacturer’s instructions. The panel includes IL-1RA, IL-10, IL-1α, IL- 1β, IL-6, IFN-α2, TNF/TNF-α, TNF-β/LT-α, sCD40L, IL-12p40, IFN-γ, IL-12/IL-12p70, IL-4, IL-5, IL-13, IL-9, IL-17A, GRO/CXCL1, IL-8/CXCL8, eotaxin/CCL11, MDC/CCL22, fractalkine/CX3CL1, IP-10/CXCL10, MCP-1/CCL2, MCP-3/CCL7, MIP-1α/CCL3, MIP-1β/CCL4, IL-2, IL-7, IL-15, GM-CSF, Flt-3L/CD135, G-CSF, IL-3, EGF, FGF-2, TGF-α, and VEGF. Fluorescence was quantified using a Luminex 200TM instrument. Cytokine/chemokine concentrations were calculated using Milliplex Analyst software version 4.2 (EMD Millipore). Luminex assay and analysis were performed by the UCLA Immune Assessment Core.

#### RNA sequencing, mapping, quantifications and quality control

Total RNA was isolated with Qiagen RNA Blood kit, and quality control was performed with Nanodrop 8000 and Agilent Bioanalyzer 2100. Globin RNA was removed with Life Technologies GLOBINCLEAR (human) kit. Libraries for RNA-Seq were prepared with KAPA Stranded mRNA-Seq Kit. The workflow consists of mRNA enrichment, cDNA generation, and end repair to generate blunt ends, A-tailing, adaptor ligation and PCR amplification. Different adaptors were used for multiplexing samples in one lane. Sequencing was performed on Illumina Hiseq3000 for a single read 50 run. Each sample generated an average of 15 million reads. Data quality check was done on Illumina SAV. Demultiplexing was performed with Illumina Bcl2fastq2 v 2.17 program.

### Computational Analysis

#### Data Normalization

Prior to analysis, data were separated into three matrices: one of RNA sequencing measurements, one of cytokine measurements from serum samples, and a third of cytokine measurements from plasma samples. These measurements included available data from all three cohorts. For both sets of cytokines measurements, values above or below the limit of detections were set to be equal to those limits. One specific cytokine - IL-12p70 - in the first cohort had a particularly low limit, so values were set to the lowest measured value to prevent biasing results during normalization. To normalize the RNA sequencing matrix, data was first mean centered and varianced scaled for each subject across all genes followed by the same process for each gene across all subjects. This was to correct for any potential batch differences before providing an accurate picture of the magnitude of gene expression changes. As each cytokine may span a different range of values, measurements were first log transformed and then mean centered first for each subject across all genes followed by the same process for each gene across all subjects. Finally, because tensor factorization attempts to explain variation among the data, we multiplied each matrix by the reciprocal of its standard deviation to ensure equal overall weighting of each.

#### Total Matrix-Tensor Factorization

We decomposed the systems serology measurements into a reduced series of Kruskal-formatted factors. Tensor operations were defined using Tensorly [@arXiv:1610.09555]. To capture the structure of the data, where the majority of measurements were made for specific antigens, but gp120-associated antibody glycosylation was measured in an antigen-generic form, we separated these two types of data into a separate 3-mode tensor and matrix, with shared subject-dimension factors:

$$X_{antigen} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

$$X_{glycosylation} \approx \sum_{r=1}^R a_r \circ d_r$$
where $a_r$, $b_r$, and $c_r$ are vectors indicating variation along the subject, receptor, and antigen dimensions, respectively. $d_r$ is a vector indicating variation along glycan forms within the glycan matrix.

Decomposition was performed through an alternating least squares (ALS) scheme [@doi:10.1137/07070111X], in the same approach as we recently reported elsewhere [@DOI:10.1101/2021.01.03.425138]. Each least squares step was performed separately for each slice along a given mode, with missing values removed. While this made each iteration step much slower, convergence was much faster as a consequence of requiring fewer iterations. Missing values did not strictly follow a tensor slice pattern, and so alternative approaches such as a sampling Khatri-Rao product were disregarded as they would still require iterative filling [@doi:10.1137/17M1112303]. This strategy required many more iterations due to a high fraction of missing values (43%). The ALS iterations were repeated until the improvement in R2X over the last ten iterations was less than $1\times 10^{-7}$.

To enforce shared factors along the subject dimension, the antigen tensor and glycan matrix were concatenated after tensor unfolding. The Khatri-Rao product of the receptor and antigen factors was similarly concatenated to the glycan factors. The least-squares solution on this axis therefore solved for minimizing the squared error across both data compendiums. The other dimensions were solved using a standard ALS approach.

<!-- TODO: Write out equations for ALS and the shared dimension. -->

#### Reconstruction Fidelity

To calculate the fidelity of our factorization results, we calculated the percent variance explained. First, the total variance was calculated by summing the variance in both the antigen-specific tensor and glycan matrix:
$$v_{total} = \left \| X_{antigen}  \right \| + \left \| X_{glycosylation}  \right \|$$
Any missing values were ignored in the variance calculation throughout. Then, the remaining variance after taking the difference between the original data and its reconstruction was calculated:
$$v_{r,antigen} = \left \| X_{antigen} - \hat X_{antigen}  \right \|$$
An analogous equation was used for the glycan matrix. Finally, the fraction of variance explained was calculated:
$$R2X = 1 - \frac{v_{r,antigen} + v_{r,glycosylation}}{v_{total}}$$
Where indicated, this quantity was calculated for values left out to assess the fidelity of imputation. In these cases this quantity was only calculated on those left out values, and indicated as Q2X.

#### Gene expression deconvolution

Following the protocol of the R package "immunedeconv" [@PMID:31510660; @PMID:32124323], we used a matrix of TPM values as input to perform the deconvolution. The read counts per gene were normalized into transcripts per kilobase million (TPM) reads mapped for the human genes. For this analysis, we considered only genes with >10 counts per million in 10 or more samples as expressed and tested a total of 12,253 human genes (out of 58,870 human genes). Deconvolution was then performed using CIBERSORT using the LM22 signatures matrix file consisting of 547 genes that accurately distinguish 22 mature human hematopoietic populations isolated from peripheral blood or in vitro culture conditions, including seven T cell types, naive and memory B cells, plasma cells, NK cells, and myeloid subsets [@PMID:31510660].  


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

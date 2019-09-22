# Improving cancer drug response prediction using tensor factorization

Identifying the molecular changes that drive cancer development has led to targeted inhibitors that selectively block these processes. These agents can be remarkably effective; however, they are usually effective only in a small subset of patients. Consequently, if we can identify features that predict which patients will respond to a targeted agent, we might vastly improve the benefit of these drugs for patients.

One strategy to identify which cancer features might predict drug response has been to take a panel of cell lines of varied origin, then profile both their molecular and drug response features. The Cancer Cell Line Encyclopedia is one such effort, wherein over one thousand cell lines have been profiled for their response to a panel of compounds, gene expression, mutations, and other molecular features. These data then make it possible to build models that attempt to predict drug response.

Molecular profiling data present unique modeling challenges. Even in these large-scale efforts, many more features are measured than cell lines, and so regularization methods must be applied. While many statistical models assume features are unrelated to one another, the molecules in a cell are not all unrelated—e.g., a mutation in one gene influences the expression of many others. As a result, many features can be strongly correlated with one another. Finally, multi-omic efforts, or projects in which different forms of molecular data (gene expression, methylation, etc) are profiled, have generally arrived at different modeling answers depending upon the type of molecular data used.

This project aims to use factorization methods, and specifically those for high-dimensional data such as that encountered with multi-omic datasets, to overcome these issues. Like principal components analysis for matrices, tensor factorization provides benefits in computational tractability, improves model performance when dealing with correlated features within a data set, and helps to visualize large, complex data. The first stage of this project will be to import and prepare the large-scale molecular profiling dataset from the Cancer Cell Line Encyclopedia. With these arranged so that they can be computationally analyzed, the data will be explored using both canonical polyadic decomposition and Tucker factorization. In the final stage of the project, with the data factored into a reduced form, these reduced features will be regressed against drug response to identify factors that predict drug response.

The objectives of this project are:

1. To import and prepare an existing large-scale dataset of cell line pharmacologic response for further analysis (Ghandi *et al*, *Nature*, 569, 503–508 (2019)).
2. Apply canonical polyadic decomposition and Tucker factorization of the data tensor of mutation, copy number, methylation, and expression data.
3. Perform regression between molecular factors and drug response.

Achieving these objectives will enable future experimental work to validate the molecular basis of the factors identified and that they predict drug response. Doing so might enable molecular measurements that predict drug response in patients.

Students will work at UCLA bioengineering research labs. A machine learning class (such as BE 188/BE 175) is highly recommended before undertaking this project.

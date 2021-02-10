"""Functions for exploring components to MRSA parafac2 decomposition"""
import copy
import pandas as pd
import gseapy as gp
from pybiomart import Server
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC


def find_SVC_proba(patient_matrix, outcomes):
    """Given a particular patient matrix and outcomes list, performs cross validation of SVC and returns the decision function to be used for AUC"""
    proba = cross_val_predict(SVC(kernel="rbf"), patient_matrix, outcomes, cv=30, method="decision_function")
    return proba


def label_points(df, names, ax):
    """Label given points given df of coordinates and names"""
    for _, point in df.iterrows():
        ax.text(point[df.columns[0]] + 0.002, point[df.columns[1]], str(point[names]), fontsize=13, fontweight="semibold", color="k")


def ensembl_convert(factors, geneids):
    """Converts array of gene weights and list of ensembl ids to dataframe for gsea"""
    # Import ensembl for id conversion
    convtable = pd.DataFrame()
    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    convtable = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
    ourids = copy.deepcopy(geneids)
    newnames = []
    newtens = pd.DataFrame(factors)
    newtens["ensembl ids"] = ourids
    # droppedids = newtens[~newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    newtens = newtens[newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    for ensid in newtens["ensembl ids"]:
        table = convtable[convtable["Gene stable ID"] == ensid]
        table.reset_index(inplace=True)
        newnames.append(table.at[0, "Gene name"])

    newtens["Gene ID"] = newnames

    return newtens


def prerank(newtens, component, geneset):
    """Runs prerank gsea on specific component/gene list"""
    prtens = pd.concat((newtens["Gene ID"], newtens[newtens.columns[component]]), axis=1)
    pre_res = gp.prerank(
        rnk=prtens, gene_sets=geneset, processes=16, min_size=1, max_size=5000, permutation_num=100, weighted_score_type=0, outdir=None, seed=6
    )
    return pre_res.res2d

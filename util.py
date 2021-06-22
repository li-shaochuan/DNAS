# coding=utf-8
#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
matplotlib.use('Agg')


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def get_diffential_gene_ttest(gene1,gene2):
    wt = gene1.mean(axis=0) #一整列求平均
    ko = gene2.mean(axis=0)
    fold =ko - wt
    pvalue = []
    for i in range(gene1.shape[1]):
        ttest = stats.ttest_ind(gene1[:, i], gene2[:, i])
        pvalue.append(ttest[1])

    qvalue = p_adjust_bh(np.asarray(pvalue))
    fold_cutoff = 1
    qvalue_cutoff = 0.05

    filtered_ids = list()
    for i in range(gene1.shape[1]):
        if(abs(fold[i]) >= fold_cutoff) and (qvalue[i] <= qvalue_cutoff):
             filtered_ids.append(i)
    return filtered_ids

def get_diffential_gene_whitney(gene1,gene2):
    wt = gene1.mean(axis=0) #一整列求平均
    ko = gene2.mean(axis=0)
    fold =ko - wt
    pvalue = []
    for i in range(gene1.shape[1]):
        u, p=stats.mannwhitneyu(gene1[:, i], gene2[:, i], alternative='two-sided')
        pvalue.append(p)

    qvalue = p_adjust_bh(np.asarray(pvalue))


    fold_cutoff = 1
    qvalue_cutoff = 0.05

    filtered_ids = list()
    for i in range(gene1.shape[1]):
        if(abs(fold[i]) >= fold_cutoff) and (qvalue[i] <= qvalue_cutoff):
             filtered_ids.append(i)
    return filtered_ids

def gene_selection(x,y):
    geneset_array=np.array(x)
    label=np.array(y).squeeze()
    CMS1 = np.where(np.array(label) == 0)
    CMS2 = np.where(np.array(label) == 1)
    CMS3 = np.where(np.array(label) == 2)
    CMS4 = np.where(np.array(label) == 3)

    CMS1_features = np.squeeze(geneset_array[CMS1, :])
    CMS2_features = np.squeeze(geneset_array[CMS2, :])
    CMS3_features = np.squeeze(geneset_array[CMS3, :])
    CMS4_features = np.squeeze(geneset_array[CMS4, :])

    # 合并
    CMS1_2_3_features = np.vstack((CMS1_features, CMS2_features, CMS3_features))
    CMS1_3_4_features = np.vstack((CMS1_features, CMS3_features, CMS4_features))
    CMS1_2_4_features = np.vstack((CMS1_features, CMS2_features, CMS4_features))
    CMS2_3_4_features = np.vstack((CMS2_features, CMS3_features, CMS4_features))
    diff_gene = list()

    diff_gene.append(get_diffential_gene_whitney(CMS4_features, CMS1_2_3_features))
    diff_gene.append(get_diffential_gene_whitney(CMS2_features, CMS1_3_4_features))
    diff_gene.append(get_diffential_gene_whitney(CMS3_features, CMS1_2_4_features))
    diff_gene.append(get_diffential_gene_whitney(CMS1_features, CMS2_3_4_features))

    diff_gene.append(get_diffential_gene_ttest(CMS4_features, CMS1_2_3_features))
    diff_gene.append(get_diffential_gene_ttest(CMS2_features, CMS1_3_4_features))
    diff_gene.append(get_diffential_gene_ttest(CMS3_features, CMS1_2_4_features))
    diff_gene.append(get_diffential_gene_ttest(CMS1_features, CMS2_3_4_features))

    diff_gene_index = list()
    for i in diff_gene:
        for j in i:
            if j not in diff_gene_index:
                diff_gene_index.append(j)

    return diff_gene_index


import numpy as np
import os
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import csv
import pandas as pd
import matplotlib.ticker as plticker
from matplotlib import colors

def createRppaTensor(filePath_rppa, fileName_rppa, filePath_sig_proteins, fileName_sig_proteins):
    rppa_file = os.path.join(filePath_rppa,fileName_rppa)
    rppa_sheets = pd.read_excel(rppa_file, sheet_name=None)
    rppa_sheet = rppa_sheets['MDD_RPPA_Level3_annotated'].set_index('Protein')
    rppa_sheet.columns = [c.split('_')[0]+'_'+c.split('_')[1] for c in rppa_sheet.columns]
    rppa_p005_file = os.path.join(filePath_sig_proteins, fileName_sig_proteins)
    rppa_p005 = pd.read_excel(rppa_p005_file, sheet_name='Sheet1').set_index('Protein')
    sig_proteins = rppa_p005.index
    rppa_sheet = rppa_sheet[rppa_sheet.index.isin(sig_proteins)]
    rppa_sheet = rppa_sheet.apply(pd.to_numeric, errors='ignore')
    rppa_averaged = rppa_sheet.groupby(rppa_sheet.columns, axis=1, sort=False).mean()
    rppa_averaged = rppa_averaged[rppa_averaged.columns[~rppa_averaged.columns.str.endswith('_1')]]
    rppa_averaged= rppa_averaged[['Ctrl_4', 'Ctrl_8', 'Ctrl_24', 'Ctrl_48', 
               'EGF_4', 'EGF_8', 'EGF_24', 'EGF_48',
               'HGF_4', 'HGF_8', 'HGF_24', 'HGF_48', 
               'OSM_4', 'OSM_8','OSM_24', 'OSM_48', 
               'BMP2_4','BMP2_8', 'BMP2_24', 'BMP2_48', 
               'IFNG_4', 'IFNG_8', 'IFNG_24','IFNG_48', 
               'TGFB_4', 'TGFB_8', 'TGFB_24', 'TGFB_48'
               ]]
    time_conditions=['_4','_8','_24','_48']
# rppa_averaged=rppa_averaged.apply(lambda x:x.columns, axis=1)
    for time_condition in time_conditions:
        time_m=rppa_averaged.columns.str.endswith(time_condition)
        conditions=rppa_averaged.columns[time_m&(~rppa_averaged.columns.str.contains('Ctrl'))].unique()
        print (conditions)
        for condition in conditions:
            rppa_averaged[condition] = rppa_averaged[condition]-rppa_averaged['Ctrl'+time_condition]
        rppa_averaged=rppa_averaged.drop('Ctrl'+time_condition,axis=1)    
    rppa_averaged_binary = rppa_averaged.copy(deep=True)
    rppa_averaged_binary[rppa_averaged_binary < 0] = -1
    rppa_averaged_binary[rppa_averaged_binary >= 0] = 1
# rppa_averaged
# rppa_averaged = rppa_averaged.sub(rppa_averaged['ctrl_0'], axis=0).drop('ctrl_0',axis=1)
    rppa_averaged = np.absolute(rppa_averaged)
# rppa_averaged = np.power(2, rppa_averaged)
    protein_names = rppa_averaged.index
# tensor = rppa_averaged.to_numpy().reshape([275,6,5]).transpose([2, 0, 1])[1:,:,:]
    tensor = rppa_averaged.to_numpy().reshape([rppa_averaged.shape[0],6,4]).transpose([2, 0, 1])
    ligand_names = pd.Index([v.split('_')[0] for v in rppa_averaged.columns.to_numpy().reshape([6,4]).transpose([1,0])[0]])
    print('Size of the RPPA tensor:',tensor.shape)
    print(ligand_names)
    return rppa_averaged, rppa_averaged_binary, tensor, protein_names, ligand_names

def createGcpTensor(filePath_gcp, fileName_gcp, filePath_sig_histones, fileName_sig_histones):
    gcppbs_file = os.path.join(filePath_gcp,fileName_gcp)
    gcp_sheet = pd.read_excel(gcppbs_file, sheet_name='two_NaN_histone_removed_T')
    gcp_sheet_T = gcp_sheet.set_index('histone')
    gcp_p005_file = os.path.join(filePath_sig_histones, fileName_sig_histones)
    gcp_p005 = pd.read_excel(gcp_p005_file, sheet_name='Sheet1').set_index('histone')
    sig_histones = gcp_p005.index
    gcp_averaged = gcp_sheet_T.groupby(gcp_sheet_T.index, sort=False).mean()
    gcp_averaged = gcp_averaged.drop(columns=gcp_averaged.columns[~gcp_averaged.columns.isin(sig_histones)])
    gcp_averaged= gcp_averaged.reindex(['ctrl_0',
               'EGF_4', 'EGF_8', 'EGF_24', 'EGF_48',
               'HGF_4', 'HGF_8', 'HGF_24', 'HGF_48', 
               'OSM_4', 'OSM_8','OSM_24', 'OSM_48', 
               'BMP2_4','BMP2_8', 'BMP2_24', 'BMP2_48', 
               'IFNG_4', 'IFNG_8', 'IFNG_24','IFNG_48', 
               'TGFB_4', 'TGFB_8', 'TGFB_24', 'TGFB_48'
               ])
    time_conditions=['_4','_8','_24','_48']
    tensor_gcp = []
    for time_condition in time_conditions:
        time_m=gcp_averaged.index.str.endswith(time_condition)
        conditions=gcp_averaged.index[time_m].unique()
        print (conditions)
        temp_ = []
        for condition in conditions:
            gcp_averaged.loc[condition] =  np.absolute(gcp_averaged.loc[condition]-gcp_averaged.loc['ctrl_0'])
            temp_.append(gcp_averaged.loc[condition].values)
        tensor_gcp.append(temp_)
    gcp_averaged = gcp_averaged.drop('ctrl_0')
    gcp_averaged_binary = gcp_averaged.copy(deep=True)
    gcp_averaged_binary[gcp_averaged_binary < 0] = -1
    gcp_averaged_binary[gcp_averaged_binary >= 0] = 1
    gcp_averaged_binary=gcp_averaged_binary[['EGF_4', 'HGF_4', 'OSM_4', 'BMP2_4', 'IFNG_4', 'TGFB_4',
       'EGF_8', 'HGF_8', 'OSM_8', 'BMP2_8', 'IFNG_8', 'TGFB_8',
       'EGF_24', 'HGF_24', 'OSM_24', 'BMP2_24', 'IFNG_24', 'TGFB_24',
       'EGF_48', 'HGF_48', 'OSM_48', 'BMP2_48', 'IFNG_48', 'TGFB_48']]
    tensor_gcp = np.array(tensor_gcp)
    print('Size of the GCP tensor:',tensor_gcp.shape)
    histone_names = gcp_averaged.columns
    ligand_names_gcp = ['EGF', 'HGF', 'OSM', 'BMP2', 'IFNG', 'TGFB']
    return gcp_averaged, gcp_averaged_binary, tensor_gcp, histone_names, ligand_names_gcp

from mpl_toolkits.mplot3d import Axes3D
def create3dTensorVisualization(tensor, x_vals, y_vals, z_vals, x_label, y_label, z_label, img_filePath, img_name):
    tensor_T = np.transpose(tensor,[1,2,0])
    print('tensor size:',tensor_T.shape)
    x,y,z = tensor_T.nonzero()
    fig = plt.figure(figsize=(9,8))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
# ax.set_yticks([0,1,2,3,4,5])
# ax.set_zticks([0,1,2,3,4])
    ax.set(yticks=range(tensor_T.shape[1]), yticklabels=y_vals,
#        xticks=range(tensor_T.shape[0]), xticklabels=protein_names)
       zticks=range(tensor_T.shape[2]), zticklabels=z_vals) 
    ax.set_xlabel(str(tensor_T.shape[0])+ x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.yaxis.labelpad=15
    ax.set_zlabel(z_label, fontsize=18)
    c = (z<=0)
    ax.scatter(x, y, z, cmap='cm.coolwarm')
    plt.show()
    plt.savefig(os.path.join(img_filePath, img_name), format="png", resolution=1200)
    return tensor_T

from numpy.linalg import norm
def compute_namda(A, B, C):
    K = A.shape[1]
    namda = 1
    for r in range(K):
        A_r = A[:,r]
        B_r = B[:,r]
        C_r = C[:,r]
        namda *= np.linalg.norm(A_r)*np.linalg.norm(B_r)*np.linalg.norm(C_r)
    return namda

def rank_(cc_values, namdas, k):
    top_50percent = int(len(cc_values)/2)
    top_idx = np.argsort(cc_values)[-top_50percent:]
    cc_values_selected = cc_values[top_idx]
    namdas_selected = namdas[top_idx]
    top_idx = np.argsort(namdas_selected)[-k:]
    return cc_values_selected[top_idx]

def rank_k(cc_values, namdas, k):
    top_50percent = int(len(cc_values)/2)
    top_idx = np.argsort(cc_values)[-k:]
    return cc_values[top_idx]

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
importr('multiway')
parafac = robjects.r["parafac"]
corcondia = robjects.r["corcondia"]
import tensorly as tl
import statistics 

def getCoreConsistency(tensor, imageName, iters = 100, num_k = 11, start =2, top_k=20):
    cc_values = {}
    cc_factors = {}
    for i in range(iters):
        for k in range(start, num_k):
            pfac = parafac(tensor, nfac=k, nstart=1)
            A_f = np.array(pfac[pfac.names.index('A')])
            B_f = np.array(pfac[pfac.names.index('B')])
            C_f = np.array(pfac[pfac.names.index('C')])
            namda = compute_namda(A_f, B_f, C_f)
            if k not in cc_values.keys():
                cc_values[k] = [corcondia(tensor, pfac, divisor = ["core", "nfac"])[0]]
                cc_factors[k] = [namda]
            else:
                cc_values[k].append(corcondia(tensor, pfac, divisor = ["core", "nfac"])[0])
                cc_factors[k].append(namda)
    ys = []
    ys_average = []
    dictionary_k_scores = dict()
    for k in range(start, num_k):
        ccvs = np.array(cc_values[k])
        ccfs = np.array(cc_factors[k])
        top_ccs = rank_k(ccvs, ccfs, top_k)
        ys.append(top_ccs)
        ys_average.append(statistics.mean(top_ccs))
        dictionary_k_scores[k] = {"mean" : statistics.mean(top_ccs), "sd" : statistics.stdev(top_ccs)}
    ys = np.array(ys)
    xs = np.array([list(range(start, num_k))] * top_k).T
    xs_average = list(range(start, num_k))
    # xs
    fig = plt.figure(figsize=(7, 4.3))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.scatter(xs, ys, 10, c='blue')
    linestyle = {"linestyle":"-", "linewidth":2, "markeredgewidth":2, "elinewidth":2, "capsize":5}
    plt.errorbar(xs_average, ys_average, color="green", **linestyle, fmt='-')    
    plt.xlabel("# of components", fontsize=16)
    plt.ylabel("Core consistency", fontsize=16)
    plt.grid(linestyle='-.')
    plt.savefig(imageName, format = "png", resolution = 1200)
    plt.show()
    sorted_dictionary_by_mean = sorted(dictionary_k_scores.items(), key= lambda m: (m[1]['mean']), reverse = True)
    pairs = pd.DataFrame([xs_average,ys_average])
    # pairs = pairs.set_index(0).rename_axis(None)
#    pairs.columns = [[''],['']]
    print("Number of K vs. Core consistency")
    return sorted_dictionary_by_mean[0][0]


from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from math import sqrt
def euclidean_norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))

def _initialize_nmf(X, n_components, init=None, eps=1e-26,
                    random_state=None):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    eps : float
        Truncate all values less then this in output to zero.
    random_state : int, RandomState instance, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape
    
    print(n_samples, n_features)

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype,
                                                             copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype,
                                                            copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = euclidean_norm(x_p), euclidean_norm(y_p)
        x_n_nrm, y_n_nrm = euclidean_norm(x_n), euclidean_norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H


import tensorly as tl
from tensorly.base import unfold
import tensorly.decomposition.candecomp_parafac as td
from tensorly.kruskal_tensor import (kruskal_to_tensor, KruskalTensor,
                              unfolding_dot_khatri_rao, kruskal_norm)

def init_nnsvd_test(tensor, n_component, random_state=None, normalize_factors=False):
    factors = td.initialize_factors(tensor, n_component, init='svd', non_negative=True,
                                    svd='numpy_svd', random_state=None, normalize_factors=None)
    
    return factors

##This function is never called?

def init_nnsvd(tensor, n_component):
    factors = []
    for mode in range(tl.ndim(tensor)):
        um = unfold(tensor, mode)
#         print('um:',um.shape)
        W, H = _initialize_nmf(um, n_component, init='nndsvd', eps=1e-6,
                random_state=None)
        factors.append(W)
    return factors

import sktensor
from sktensor import dtensor
import ncp
from ncp import nonnegative_tensor_factorization

def factorizeNCP(tensor, components):
    num_component = components
    X = dtensor(tensor) 
# print(X.shape)
    init_factors = init_nnsvd(X, num_component)
# print(init_factors[0])
# np.random.seed(4)
    X_approx_ks = ncp.nonnegative_tensor_factorization(X, num_component, method='anls_bpp', 
                                                   init=init_factors.copy())

    C = X_approx_ks.U[0]
    A = X_approx_ks.U[1]
    B = X_approx_ks.U[2]
    print('[A,B,C]:', A.shape, B.shape, C.shape)
    return A, B, C

import tensorly as tl
from tensorly.decomposition import non_negative_parafac
def factorizeTensorly(tensor, components):
    num_component = 2
    print('tensor{}'.format(tensor.shape),'= component_1 + component_2 + ... + component_{}= [A,B,C]'.format(num_component))
    weights, factors = non_negative_parafac(tensor, rank=num_component, init='svd')
    C = np.array(factors[0])
    A = np.array(factors[1])
    B = np.array(factors[2])
    print('[A,B,C]:', A.shape, B.shape, C.shape)
    print(C)
    print(B)
    return A, B, C 

def factorize(tensor, components, method="ncp"):
    if method == "ncp":
        return factorizeNCP(tensor, components)
    else:
        return factorizeTensorly(tensor, components)

def plotFactorMatrices(A, B, C, tensor, imgName_A, imgName_B, imgName_C, ligand_names):
    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.matshow(A)
    ax1.set_aspect('auto')
    ax1.set_title('A', y=-0.1)
    fig.colorbar(cax1)
    ax1.set_ylabel(str(tensor.shape[1])+' proteins', size=(13))
    ax1.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    #plt.show()
    plt.savefig(imgName_A, format="png", resolution=1200)
    ax2 = fig.add_subplot(132)
    cax2 = ax2.matshow(B)
    ax2.set_aspect('auto')
    ax2.set_title('B', y=-0.1)
    fig.colorbar(cax2)
# ax2.set_ylabel(str(tensor.shape[2])+" ligands", size=(13))
    ax2.set_yticklabels(ligand_names.insert(0,''))
    ax2.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    #plt.show()
    plt.savefig(imgName_B, format="png", resolution=1200)
    ax3 = fig.add_subplot(133)
    cax3 = ax3.matshow(C)
    ax3.set_aspect('auto')
    ax3.set_title('C', y=-0.1)
# ax3.set_ylabel(tensor.shape[0], size=(13))
# ax3.set_yticklabels(['','4_hour','8_hour','24_hour','48_hour'])
    ax3.set_yticklabels(['','4_hour','8_hour','24_hour','48_hour'])
    ax3.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    fig.colorbar(cax3)
    plt.show()
    plt.savefig(imgName_C, format="png", resolution=1200)
    fig.colorbar(cax3)

def componentPredictionsForFactors(A, B):
    component_preds = A.argmax(axis=1) # classify proteins into respective components based on the largest membership values
    print('{} proteins in total, each protein belongs to a predicted component with the largest membership value:\n\n'.format(A.shape[0]), component_preds)
    component_preds_B = B.argmax(axis=1) # classify proteins into respective components based on the largest membership values
    print('{} proteins in total, each protein belongs to a predicted component with the largest membership value:\n\n'.format(B.shape[0]), component_preds_B)
    return component_preds, component_preds_B

def getClusterMembershipProbabilityA(A,component_preds,  rppa_averaged_binary, protein_names, img_title, y_label, num_component, imgName, rppa_averaged):
    new_indexes = []
    clusters = []
    for k in range(num_component):
        idx = np.where(component_preds == k)[0]
        new_indexes.extend(idx)
        clusters.append(idx)
    cluster_id_ranges = []
    count = 0
    for clu in clusters:
        id_range = str(count) + '~' + str(count+len(clu)-1)
        cluster_id_ranges.append(id_range)
        count += len(clu)
    print(cluster_id_ranges)
# rank communities as a diagonal matrix
    A_ranked = A[new_indexes]
    protein_names_ranked = protein_names[new_indexes]
    tensor_binary = rppa_averaged_binary.to_numpy()[new_indexes].reshape([rppa_averaged.shape[0],6,4]).transpose([1, 0, 2])
    print(tensor_binary.shape)
    fig = plt.figure(figsize=(10,7))
# fig.subplots_adjust(hspace=0, wspace=0.3)
    ax = fig.add_subplot(111)
    aa = ax.matshow(A_ranked,cmap=plt.cm.gray_r)
    ax.set_aspect('auto')
    ax.set_title('Components')
    ax.set_ylabel(y_label, size=(13))
    ax.set_xticks(range(0,num_component))
    fig.colorbar(aa)
    plt.savefig(imgName, format="png", resolution=1200)
    return tensor_binary, A_ranked, protein_names_ranked, clusters

def getClusterMembershipProbability(factor_matrix,components, labels, img_title, y_label, rank, imgName):
    num_component = rank
    # rerange the proteins in matrix A
    new_indices = []
    clusters = []
    for k in range(num_component):
        idx = np.where(components == k)[0]
        new_indices.extend(idx)
        clusters.append(idx)
    cluster_id_ranges = []
    count = 0
    for clu in clusters:
        id_range = str(count) + '~' + str(count+len(clu)-1)
        cluster_id_ranges.append(id_range)
        count += len(clu)
    print(cluster_id_ranges)  
# rank communities as a diagonal matrix
    factor_matrix_ranked = factor_matrix[new_indices]
    labels_ranked = labels[new_indices]
    fig = plt.figure(figsize=(10,7))
# fig.subplots_adjust(hspace=0, wspace=0.3)
    #logging.debug("Plotting the factor matrix and labels.. ")
    ax = fig.add_subplot(111)
    aa = ax.matshow(factor_matrix_ranked,cmap=plt.cm.gray_r)
    ax.set_aspect('auto')
    ax.set_title(img_title)
    ax.set_ylabel(y_label, size=(13))
    ax.set_xticks(range(0,rank))
    fig.colorbar(aa)
    fig.savefig(imgName, format="png", resolution=1200)
    return factor_matrix_ranked, clusters, labels_ranked


def plotLatentFactor(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, img_filePath):
    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    ax1 = fig.add_subplot(131)
    plt.set_cmap('PiYG')
    norm = colors.DivergingNorm(vcenter=0)
    cax1 = ax1.matshow(A_ranked,norm=norm)
    ax1.set_aspect('auto')
    ax1.set_title('$\mathbf{A}^r$', y=-0.1, fontsize=15)
    plt.rc('ytick',labelsize=12)
    fig.colorbar(cax1)
    plt.savefig(os.path.join(img_filePath, "A.png"), format="png", resolution=1200)
# ax1.set_ylabel(str(tensor.shape[1])+' proteins', fontsize=14)
    ax1.set_yticks(range(len(protein_names_ranked)))
    loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
    ax1.yaxis.set_major_locator(loc)
    show_protein_names = protein_names_ranked[range(0, len(protein_names_ranked), 10)].insert(0,'')
    ax1.set_yticklabels(show_protein_names, fontsize=11)
    ax1.set_xticklabels(['','comp_1', 'comp_2', 'comp_3'], fontsize=12)
    ax2 = fig.add_subplot(132)
    plt.set_cmap('PiYG')
    cax2 = ax2.matshow(B_ranked,norm =colors.DivergingNorm(vcenter=0))
    ax2.set_aspect('auto')
    ax2.set_title('$\mathbf{B}^r$', y=-0.1, fontsize=15)
    fig.colorbar(cax2)
    plt.rc('ytick',labelsize=12)
    plt.savefig(os.path.join(img_filePath, "B.png"), format="png", resolution=1200)
# ax2.set_ylabel(str(tensor.shape[2])+" ligands", size=(13))
    ax2.set_yticklabels(ligand_names_ranked.insert(0,''))
    ax2.set_xticklabels(['','comp_1', 'comp_2', 'comp_3'], fontsize=12)
    ax3 = fig.add_subplot(133)
    cax3 = ax3.matshow(C, cmap='PiYG', norm =colors.DivergingNorm(vcenter=0))
    ax3.set_aspect('auto')
    ax3.set_title('$\mathbf{C}^r$', y=-0.1, fontsize=15)
    plt.rc('ytick',labelsize=12)
# ax3.set_ylabel(tensor.shape[0], size=(13))
    ax3.set_yticklabels(['', '4hr','8hr','24hr','48hr'])
    ax3.set_xticklabels(['','comp_1', 'comp_2', 'comp_3'], fontsize=12)
    plt.savefig(os.path.join(img_filePath, "C.png"), format="png", resolution=1200)
    plt.show()
    fig.colorbar(cax3)

def saveFactorsToExcel(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, filePath, fileName):
    A_out = pd.DataFrame(data=A_ranked, index=protein_names_ranked, columns=['com_1','com_2'])
    B_out = pd.DataFrame(data=B_ranked, index=ligand_names_ranked, columns=['com_1','com_2'])
    C_out = pd.DataFrame(data=C, index=['4_hour','8_hour','24_hour','48_hour'], columns=['com_1','com_2'])
    writer = pd.ExcelWriter(os.path.join(filePath, fileName))
    A_out.to_excel(writer, "A")
    B_out.to_excel(writer, "B")
    C_out.to_excel(writer, "C")
    writer.save()

def proteinsPerCluster(clusters, protein_names):
    for k in range(len(clusters)):
        proteins = ' '.join(protein_names[clusters[k]])
    print('Protein cluster {}: {}\n'.format(k, proteins))
    return proteins

from matplotlib import colors
def plotClusters(factor_matrix, clusters, labels,imgName):
    fig = plt.figure(figsize=(3,4))
    colors = ['darkorange','deepskyblue']
    for k in range(len(clusters)):
        plt.scatter(factor_matrix[clusters[k], 0], factor_matrix[clusters[k], 1], c=colors[k], s=20,label='Cluster '+str(k+1))
    show_ids=[]
    for v in np.random.choice(clusters[0],2,replace=False):
        if factor_matrix[v, 0] > 0.6:
            plt.text(factor_matrix[v, 0], factor_matrix[v, 1]+.03, labels[v], fontsize=8)
            show_ids.append(v)
    for v in np.random.choice(clusters[1],1,replace=False):
        if factor_matrix[v, 1] > 0.6:
            plt.text(factor_matrix[v, 0], factor_matrix[v, 1]+.03, labels[v], fontsize=8)
            show_ids.append(v)       
    plt.xticks(fontsize=13)
    plt.xlabel('Comp1', fontsize=13)
    plt.ylabel('Comp2', fontsize=13)
    plt.legend(fontsize=13)
    plt.show()
    print(show_ids)    
    fig.savefig(imgName, format="png", resolution=1200)
    return show_ids

from scipy.special import softmax
def plotProbabilityDistributionOfClusters(factor_matrix, show_ids, y_labels_ranked, img_filePath,img_name):
    #logging.debug("Plotting probability distribution for the factor matrix.. ")
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(111)
    cax1 = ax1.matshow(softmax(factor_matrix,axis=1),cmap='bwr',norm = colors.TwoSlopeNorm(vcenter=0))
    ax1.set_aspect('auto')
    ax1.set_title('Probability distribution of clusters', y=-0.1,fontsize=13)
    fig.colorbar(cax1)
    plt.rc('ytick',labelsize=12)        
    loc = plticker.MultipleLocator(base=4) # this locator puts ticks at regular intervals
    ax1.yaxis.set_major_locator(loc)
    show_y_names = y_labels_ranked[range(0, len(y_labels_ranked), 4)].to_list()
# ax1.set_yticks(range(len(protein_names_ranked)))
# # loc = plticker.MultipleLocator(base=3) # this locator puts ticks at regular intervals
# # ax1.yaxis.set_major_locator(loc)
# show_protein_names = protein_names_ranked.to_list()
    print(show_y_names)
    for i in range(len(show_y_names)):
        if i not in show_ids:
            show_y_names[i] = ''
    show_y_names = [''] + show_y_names
    ax1.set_yticklabels(show_y_names, fontsize=11)
    ax1.set_xticklabels(['','Cluster 1', 'Cluster 2', 'comp_3'],fontsize=12)
    plt.show()
    plt.savefig(os.path.join(img_filePath, img_name), format="png", resolution=1200)

def plotForFactorMatrix(factor_matrix, clusters, labels_ranked, imgName_cluster_scatter, img_filePath ,img_name_prob):
    show_ids = plotClusters(factor_matrix, clusters, labels_ranked, os.path.join(img_filePath, imgName_cluster_scatter))
    plotProbabilityDistributionOfClusters(factor_matrix, show_ids, labels_ranked, img_filePath, img_name_prob)

## Add a wrapper function to process each factor matrix
''''def processFactorMatrix(factor_matrix, tensor_dim_idx, title_matrix, label_matrix, labels_entities, img_title, rank, imgName_matrix , imgName_cluster):
    logging.debug("inside factor matrix processing wrapper function.. ")
    plotFactorMatrices(factor_matrix, tensor_dim_idx, title_matrix, label_matrix, imgName_matrix)
    components = classifyEntities(factor_matrix)
    factor_matrix_ranked, clusters, labels_ranked = getClusterMembershipProbability(factor_matrix,components, labels_entities, img_title, y_label, rank, imgName_cluster)
    plotLatentFactor(factor_matrix,  title_matrix)
    proteins_per_cluster = proteinsPerCluster(clusters, labels_entities)
    show_ids = plotClusters(factor_matrix, clusters, labels_entities,imgName_cluster)
    plotProbabilityDistributionOfClusters(factor_matrix, show_ids, labels_ranked)
    logging.debug("exiting factor matrix processing wrapper function.. ")
    return factor_matrix_ranked, clusters, labels_ranked '''

def findCorrelationMatrix(factor_matrix1, factor_matrix2):
    #logging.debug("Creating a correlation matrix for the given factor matrices.")
    B_expand = np.expand_dims(factor_matrix2, axis=1)
    patterns = np.sum(np.transpose(np.multiply(factor_matrix1, B_expand), [1,0,2]), axis=2)
    print(patterns.shape)
    return patterns

def plotCorrelationMatrix(patterns, x_labels, y_labels, img_title, img_file):
    #logging.debug("Plotting the correlation matrix.. ")
    fig = plt.figure(figsize=(2,6))
    ax = fig.add_subplot(111)
    aa = ax.matshow(patterns,cmap='bwr', norm =colors.TwoSlopeNorm(vcenter=0))
    ax.set_aspect('auto')
# ax.set_title('Components')
# ax.set_ylabel(str(tensor.shape[1])+' proteins', size=(16))
    ax.set_yticks(range(len(y_labels)))
    loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    #show_protein_names = protein_names_ranked[range(0, len(protein_names_ranked), 10)].insert(0,'')
    ax.set_yticklabels(y_labels, fontsize=11)
    plt.rc('ytick',labelsize=13)
    ax.set_xticklabels(x_labels, fontsize=11, rotation =55)
    ax.set_title(img_title, y=-0.1, fontsize=16)
    fig.colorbar(aa)
    plt.show()
    fig.savefig(img_file, format="png", resolution=1200)

from scipy.stats import norm
def plotCorrelationMatrixGaussianDistribution(patterns, xlabel, ylabel, img_title, img_file):
    fig = plt.figure(figsize=(7, 4.3))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=14)
    flatten_patterns = patterns.flatten()
    flatten_patterns.sort()
    mean, std = norm.fit(flatten_patterns)
    print('mean:{}, std:{}'.format(mean, std))
    fitted_pdf = norm.pdf(flatten_patterns, loc=mean, scale=std)
    linestyle = {"linestyle":"-", "linewidth":2, "markeredgewidth":2, "elinewidth":2, "capsize":5}
    plt.errorbar(flatten_patterns, fitted_pdf, color="green", **linestyle, fmt='-')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend([img_title], fontsize=13)
    plt.grid(linestyle='-.')
    plt.show()
    plt.savefig(img_file, format="png", resolution=1200)
    return mean, std, flatten_patterns, fitted_pdf

def writeResultsToExcel(filePath, fileName, flatten_patterns, fitted_pdf, mean, std, sheet_name):
    writer_intensity = pd.ExcelWriter(fileName)
    intense_density = np.concatenate([flatten_patterns, fitted_pdf]).reshape(2,flatten_patterns.shape[0]).transpose([1,0])
    intense_density_mat = pd.DataFrame(data=intense_density, columns=['Intensity','Density'])
    intense_density_mat.loc[-1] = [mean, std]
    intense_density_mat.index = intense_density_mat.index + 1
    intense_density_mat = intense_density_mat.sort_index()
    intense_density_mat.rename(index = {0: "mean/std"}, inplace = True)
    intense_density_mat.to_excel(writer_intensity, sheet_name)
    writer_intensity.save()

def plotCorrelationsPerLigand(patterns_AC, tensor_binary, ligand_names_ranked, img_FilePath, filePath):
    '''C_expand = np.expand_dims(factor_C, axis=1)
    patterns_AC = np.sum(np.transpose(np.multiply(factor_A_ranked, C_expand), [1,0,2]), axis=2)
    print(patterns_AC.shape) '''
    patterns_AC_by_lignad = tensor_binary * patterns_AC
    norm = colors.DivergingNorm(vcenter=0)
    fig = plt.figure(figsize=(16,10))
    plt.set_cmap('bwr')
    for i in range(patterns_AC_by_lignad.shape[0]):
        ax = fig.add_subplot(2,3,i+1)
        aa = ax.matshow(patterns_AC_by_lignad[i], norm=norm)
        ax.set_aspect('auto')
        ax.set_ylabel(str(tensor.shape[1])+' proteins', size=(13))
        ax.set_xticklabels(['','4hr','8hr','24hr','48hr'], fontsize=13)
        ax.set_title('M2_'+str(ligand_names_ranked[i]), y=-0.1)
        fig.colorbar(aa)
        plt.show()
        plt.savefig(os.path.join(img_FilePath, 'M2_'+str(ligand_names_ranked[i]+'.png')), format="png", resolution=1200)
# fitting the data with a gaussian distribution
    from scipy.stats import norm
    intensity_dist_rppa_mats = []
    fig = plt.figure(figsize=(16,10))
    plt.set_cmap('bwr')
    for i in range(patterns_AC_by_lignad.shape[0]):
        ax = fig.add_subplot(2,3,i+1)
        flatten_AC = patterns_AC_by_lignad[i].flatten()
        flatten_AC.sort()
        mean, std = norm.fit(flatten_AC)
        print('{}, mean:{}, std:{}'.format(ligand_names_ranked[i], mean, std))
        fitted_pdf = norm.pdf(flatten_AC, loc=mean, scale=std)
        writeResultsToExcel(filePath, flatten_AC, fitted_pdf, mean, std, "M2_AC_per_ligand")
        linestyle = {"linestyle":"-", "linewidth":2, "markeredgewidth":2, "elinewidth":2, "capsize":5}
        plt.errorbar(flatten_AC, fitted_pdf, color="green", **linestyle, fmt='-')
        plt.xlabel("Interaction Intensity", fontsize=13)
        plt.ylabel("Density Value", fontsize=13)
        plt.legend(['PDF (M2_'+str(ligand_names_ranked[i])+")"])
        plt.grid(linestyle='-.')
        plt.show()
        plt.savefig(os.path.join(img_FilePath, 'PDF (M2_'+str(ligand_names_ranked[i]+'.png')), format="png", resolution=1200)
    


def computeCorrelationForfactorMatrices(factor_matrix1, factor_matrix2, xlabels_plot, ylabels_plot, xlabel_pdc, ylabel_pdc,
                                        imgtitle_plot, imgtitle_pdc, imgfilePathPlot, imgfilePathPd,
                                       sheet_name, excel_path, file_name_excel):
    patterns =  findCorrelationMatrix(factor_matrix1, factor_matrix2)
    plotCorrelationMatrix(patterns, xlabels_plot, ylabels_plot, imgtitle_plot, os.path.join(imgfilePathPlot, str(imgtitle_plot+".png")))
    mean, std, flatten_patterns, fitted_pdf = plotCorrelationMatrixGaussianDistribution(patterns, xlabel_pdc, ylabel_pdc, imgtitle_pdc, os.path.join(imgfilePathPd, str(imgtitle_pdc + ".png")))
    writeResultsToExcel(excel_path, file_name_excel, flatten_patterns, fitted_pdf, mean, std, sheet_name)
    return patterns,mean,std 

def getCorrelationsForAllFactors(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, excel_path, file_name_excel, imgFilePath, imgtitles_plot, imgtitle_pdc):

    patterns_AB, mean_AB, std_AB = computeCorrelationForfactorMatrices(A_ranked, B_ranked, protein_names_ranked,ligand_names_ranked,"Intensity", "Density",
                                        imgtitles_plot[0], imgtitle_pdc[0], imgFilePath, imgFilePath,
                                       "M1_AB", excel_path,file_name_excel)
    patterns_AC, mean_AC, std_AC = computeCorrelationForfactorMatrices(A_ranked, C, protein_names_ranked, ['4_hrs', '8_hrs', '24_hrs', '48_hrs'], "Intensity", "Density",
                                        imgtitles_plot[1], imgtitle_pdc[1], imgFilePath, imgFilePath,
                                       "M2_AC", excel_path,file_name_excel)
    patterns_BC, mean_BC, std_BC = computeCorrelationForfactorMatrices(B_ranked, C, ligand_names_ranked, ['4_hrs', '8_hrs', '24_hrs', '48_hrs'], "Intensity", "Density",
                                        imgtitles_plot[2], imgtitle_pdc[2], imgFilePath, imgFilePath,
                                       "M3_BC", excel_path,file_name_excel)
    return [patterns_AB, patterns_AC, patterns_BC], [mean_AB, mean_AC, mean_BC], [std_AB, std_AC, std_BC]

def writeCorrelationsToExcel(patterns_list, sheet_names_patterns_list, filePath, protein_names_ranked, ligand_names_ranked, histone_names_ranked):
    num_protein = patterns_list[0].shape[0]
    patterns_AB_pairs = pd.DataFrame(data=patterns_list[0], index=protein_names_ranked, columns=ligand_names_ranked)
    patterns_AC_pairs = pd.DataFrame(patterns_list[1], index=protein_names_ranked, 
                              columns=histone_names_ranked) 
    num_ligand = patterns_BC.shape[0]
    patterns_BC_pairs = pd.DataFrame(data=patterns_list[2], index=ligand_names_ranked, 
                      columns=histone_names_ranked)
    writer = pd.ExcelWriter(filePath)
    patterns_AB_pairs.to_excel(writer, sheet_names_patterns_list[0])
    patterns_AC_pairs.to_excel(writer, sheet_names_patterns_list[1])
    patterns_BC_pairs.to_excel(writer, sheet_names_patterns_list[2])
    writer.save()
    return patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs

def getSignificantEntitiesForCutOff(patterns, mean, cutoff, sheet_name, filePath):
    writer_mean = pd.ExcelWriter(filePath)
    stack = patterns.T.stack().to_frame()
    stack.columns = ["Intensity"]
    stack_sorted = stack[~((stack.values-mean < cutoff))]
    stack_sorted.to_excel(writer_mean, sheet_name)
    writer_mean.save()

def getSignificantEntitiesForCenterElbow(patterns_list, cutoffs_center_elbow_list, mean_list, filePath, sheet_names_list):
    #logging.debug("Finding significant entities using the elbow of pdf curve ..")
    for i in range(len(patterns_list)):
        getSignificantEntitiesForCutOff(patterns_list[i], mean_list[i], cutoffs_center_elbow_list[i], sheet_names_list[i], filePath)

def getSignificantEntitiesForElbow(patterns_list, cutoffs_elbow_list, mean_list, filePath, sheet_names_list):
    #logging.debug("Finding significant entities using the elbow of pdf curve ..")
    for i in range(len(patterns_list)):
        getSignificantEntitiesForCutOff(patterns_list[i], mean_list[i], cutoffs_elbow_list[i], sheet_names_list[i], filePath)

def getSignificantEntitiesAsymptotic(patterns_list, cutoffs_asymptotic_list, mean_list, filePath, sheet_names_list):
    #logging.debug("Finding significant entities using the asymptote of pdf curve ..")
    for i in range(len(patterns_list)):
        getSignificantEntitiesForCutOff(patterns_list[i], mean_list[i], cutoffs_asymptotic_list[i], sheet_names_list[i], filePath)

def getSignificantEntities(patterns_list, cutoffs_elbow_list, cutoffs_center_elbow_list, cutoffs_asymptotic_list, mean_list, sheet_names_elbow, sheet_names_center_elbow, sheet_names_asymptotic, filePath):
    #logging.debug("In wrapper function to find sognificant entities..")
    getSignificantEntitiesForCenterElbow(patterns_list, cutoffs_center_elbow_list, mean_list, filePath, sheet_names_center_elbow)
    getSignificantEntitiesForElbow(patterns_list, cutoffs_elbow_list, mean_list, filePath, sheet_names_elbow)
    getSignificantEntitiesAsymptotic(patterns_list, cutoffs_asymptotic_list, mean_list, filePath, sheet_names_asymptotic)


def calculateHOCScores(patterns_AB, patterns_rppa_gcp, filePath, hoc_file_name, hoc_filtered_file_name, all_hoc_score_file_name):
    sig_ligands = ligand_names_ranked
    sig_proteins = ['CCNB1','PLK1','RB1pS807S811','RPS6pS240S244','DUSP4','RPS6pS235S236',
                'CDC2','MYH2pS1943']
    sig_histones = ['H3K9me2S10ph1K14ac0 ','H3K9me3S10ph1K14ac0 ','H3K18ub1K23ac0 ']
    writer_hoc_score = pd.ExcelWriter(hoc_file_name)
    filtered_writer_hoc_score = pd.ExcelWriter(hoc_filtered_file_name)
    for i in range(patterns_AB.shape[1]):
        ligand_i = np.expand_dims(patterns_AB[:,i], axis=0)
        patterns_rppa_gcp_i = patterns_rppa_gcp * ligand_i.T
        hoc_score_i = pd.DataFrame(data=patterns_rppa_gcp_i, index= protein_names_ranked, columns=histone_names_ranked)
#     hoc_score_i.to_excel(writer_hoc_score, ligand_names_ranked[i])
        filtered_hoc_score_i = hoc_score_i.loc[sig_proteins, sig_histones]
        filtered_hoc_score_i = filtered_hoc_score_i/filtered_hoc_score_i.max().max()
        filtered_hoc_score_i.to_excel(filtered_writer_hoc_score, ligand_names_ranked[i])
    # writer_hoc_score.save()
    filtered_writer_hoc_score.save()
    all_hoc_score_file = os.path.join(filePath, hoc_file_name)
    sig_proteins = ['CCNB1','PLK1','RB1pS807S811','RPS6pS240S244','DUSP4','RPS6pS235S236']
# -----------------------Category 3
    hoc_dict_c3 = {}
    all_hoc_score_EGF = pd.read_excel(all_hoc_score_file, sheet_name='EGF').set_index('Protein')
    filtered_hoc_score_EGF_h4 = all_hoc_score_EGF.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_EGF_h4.max())
    filtered_hoc_score_EGF_h4 = filtered_hoc_score_EGF_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<EFG,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_EGF_h4.loc[pro]
    all_hoc_score_HGF = pd.read_excel(all_hoc_score_file, sheet_name='HGF').set_index('Protein')
    filtered_hoc_score_HGF_h4 = all_hoc_score_HGF.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_HGF_h4.max())
    filtered_hoc_score_HGF_h4 = filtered_hoc_score_HGF_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<HGF,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_HGF_h4.loc[pro]
    all_hoc_score_OSM = pd.read_excel(all_hoc_score_file, sheet_name='OSM').set_index('Protein')
    filtered_hoc_score_OSM_h4 = all_hoc_score_OSM.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_OSM_h4.max())
    filtered_hoc_score_OSM_h4 = filtered_hoc_score_OSM_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<OSM,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_OSM_h4.loc[pro]
    all_hoc_score_BMP2 = pd.read_excel(all_hoc_score_file, sheet_name='BMP2').set_index('Protein')
    filtered_hoc_score_BMP2_h4 = all_hoc_score_BMP2.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_BMP2_h4.max())
    filtered_hoc_score_BMP2_h4 = filtered_hoc_score_BMP2_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<BMP2,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_BMP2_h4.loc[pro]

    all_hoc_score_IFNG = pd.read_excel(all_hoc_score_file, sheet_name='IFNG').set_index('Protein')
    filtered_hoc_score_IFNG_h4 = all_hoc_score_IFNG.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_IFNG_h4.max())
    filtered_hoc_score_IFNG_h4 = filtered_hoc_score_IFNG_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<IFNG,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_IFNG_h4.loc[pro]

    all_hoc_score_TGFB = pd.read_excel(all_hoc_score_file, sheet_name='TGFB').set_index('Protein')
    filtered_hoc_score_TGFB_h4 = all_hoc_score_TGFB.loc[sig_proteins, 'H4(20to23)K20me0']
    print(filtered_hoc_score_TGFB_h4.max())
    filtered_hoc_score_TGFB_h4 = filtered_hoc_score_TGFB_h4/1380.100965754926
    for pro in sig_proteins:
        ke = '<TGFB,' + pro + ',' + 'H4(20to23)K20me0 >'
        hoc_dict_c3[ke] = filtered_hoc_score_TGFB_h4.loc[pro]
    hoc_score_file = os.path.join('dataset/PLOS_results_2021-10-27/','HOC_scores_filtered_2.xlsx')        
# hoc_score_file = os.path.join('dataset/PLOS_results_2021-10-27/','HOC_scores_filtered.xlsx')# filtered_2 : Not found. File name 
                                                                                            # generated is filtered.xlsx     
    EGF_hoc = pd.read_excel(hoc_score_file, sheet_name='EGF').set_index('Protein')
    HGF_hoc = pd.read_excel(hoc_score_file, sheet_name='HGF').set_index('Protein')
    OSM_hoc = pd.read_excel(hoc_score_file, sheet_name='OSM').set_index('Protein')
    BMP2_hoc = pd.read_excel(hoc_score_file, sheet_name='BMP2').set_index('Protein')
    IFNG_hoc = pd.read_excel(hoc_score_file, sheet_name='IFNG').set_index('Protein')
    TGFB_hoc = pd.read_excel(hoc_score_file, sheet_name='TGFB').set_index('Protein')
    print(EGF_hoc.max().max())
    print(HGF_hoc.max().max())
    print(OSM_hoc.max().max())
    print(BMP2_hoc.max().max())
    print(IFNG_hoc.max().max())
    print(TGFB_hoc.max().max())
    hoc_score_file = os.path.join('dataset/PLOS_results_2021-10-27/','HOC_scores_filtered_2.xlsx')
# hoc_score_file = os.path.join('dataset/PLOS_results_2021-10-27/','HOC_scores_filtered.xlsx')
    EGF_hoc = pd.read_excel(hoc_score_file, sheet_name='EGF').set_index('Protein')
    HGF_hoc = pd.read_excel(hoc_score_file, sheet_name='HGF').set_index('Protein')
    OSM_hoc = pd.read_excel(hoc_score_file, sheet_name='OSM').set_index('Protein')
    BMP2_hoc = pd.read_excel(hoc_score_file, sheet_name='BMP2').set_index('Protein')
    IFNG_hoc = pd.read_excel(hoc_score_file, sheet_name='IFNG').set_index('Protein')
    TGFB_hoc = pd.read_excel(hoc_score_file, sheet_name='TGFB').set_index('Protein')

# -----------------------Category 1
    hoc_dict_c1 = {}
    sig_proteins = ['CCNB1','PLK1','RB1pS807S811','RPS6pS240S244','DUSP4','RPS6pS235S236']
    sig_histones = ['H3K9me2S10ph1K14ac0 ','H3K9me3S10ph1K14ac0 ']
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<EFG,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = EGF_hoc.loc[pro][his]/1380.100965754926
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<HGF,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = HGF_hoc.loc[pro][his]/1380.100965754926
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<OSM,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = OSM_hoc.loc[pro][his]/1380.100965754926
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<BMP2,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = BMP2_hoc.loc[pro][his]/1380.100965754926
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<IFNG,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = IFNG_hoc.loc[pro][his]/1380.100965754926
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<TGFB,' + pro + ',' + his + '>'
            hoc_dict_c1[ke] = TGFB_hoc.loc[pro][his]/1380.100965754926

# -----------------------Category 2        
    hoc_dict_c2 = {}
    sig_proteins = ['CCNB1','PLK1','RB1pS807S811','RPS6pS240S244','DUSP4','RPS6pS235S236',
                'CDC2','MYH2pS1943']
    sig_histones = ['H3K18ub1K23ac0 ']
    for pro in sig_proteins:
        for his in sig_histones:
            ke = '<EFG,' + pro + ',' + his + '>'
            hoc_dict_c2[ke] = EGF_hoc.loc[pro][his]/1380.100965754926
        
# -----------------------Category 3
# hoc_dict_c3

    all_hoc_score_group = pd.ExcelWriter(all_hoc_score_file_name)
    hoc_score_g1 = pd.DataFrame(data=hoc_dict_c1.values(), index= hoc_dict_c1.keys(), columns=['HOC_scores'])
    hoc_score_g2 = pd.DataFrame(data=hoc_dict_c2.values(), index= hoc_dict_c2.keys(), columns=['HOC_scores'])
    hoc_score_g3 = pd.DataFrame(data=hoc_dict_c3.values(), index= hoc_dict_c3.keys(), columns=['HOC_scores'])
    hoc_score_g1.to_excel(all_hoc_score_group, 'Fig10b')
    hoc_score_g2.to_excel(all_hoc_score_group, 'Fig10c')
    hoc_score_g3.to_excel(all_hoc_score_group, 'Fig10d')
    all_hoc_score_group.save()
    hoc_score_i.loc[sig_proteins, sig_histones]
    print(patterns_AB.shape)
    print(protein_names_ranked)
    print(ligand_names_ranked)
    print(patterns_rppa_gcp.shape)
    print(protein_names_ranked)
    print(histone_names_ranked)


### Creating wrapper functions for rppa , gcp and rppa_gcp_hocmo

def callRppaDataProcessingPipeline(factorization_method, filePath_rppa, fileName_rppa, filePath_sig_proteins, fileName_sig_proteins, img_FilePath, results_filePath):
    rppa_averaged, rppa_averaged_binary, tensor, protein_names, ligand_names = createRppaTensor(filePath_rppa, fileName_rppa, filePath_sig_proteins, fileName_sig_proteins)
    tensor_T = create3dTensorVisualization(tensor, [], ligand_names, ['4','8','24','48'], 'Proteins', 'Ligands', 'Time', img_FilePath, "rppa_tensor_3d.png")
    components = getCoreConsistency(tensor, os.path.join(img_FilePath, "core_consistency_rppa.png"), iters = 100, num_k = 11, start =2, top_k=20)
    print("k-main :", components)
    A, B, C = factorize(tensor, components, method=factorization_method)
    plotFactorMatrices(A, B, C, tensor, 'A_rppa.png', 'B_rppa.png', 'C_rppa.png', ligand_names)
    component_preds, component_preds_B = componentPredictionsForFactors(A, B)
    tensor_binary, A_ranked, protein_names_ranked, clusters = getClusterMembershipProbabilityA(A, component_preds,  rppa_averaged_binary, protein_names, 
                                                        'Components', 'Proteins', components, "A_ranked_per_cluster.png", rppa_averaged)
    B_ranked, clusters_B, ligand_names_ranked = getClusterMembershipProbability(B, component_preds_B, ligand_names, 'Components', 'Ligands', components, "B_ranked_per_cluster.png")
    plotLatentFactor(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, img_FilePath)
    saveFactorsToExcel(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, results_filePath, "Factors_rppa_ncp.xlsx")
    proteins = proteinsPerCluster(clusters, protein_names)
    plotForFactorMatrix(A_ranked, clusters, protein_names_ranked, "proteins_per_cluster_scatter_plot.png", img_FilePath ,"proteins_per_cluster_probability.png")
    excel_file_path = "/storage1/fs1/reetika/TME/"
    patterns_list, means_list, std_list = getCorrelationsForAllFactors(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, excel_file_path, "Intensity_density_ncp_rppa.xlsx", img_FilePath, [ "M1r", "M2r", "M3r"],
                                        ['Probability Density Function M1r', 'Probability Density Function M2r','Probability Density Function M3r'])
                                       
    patterns_AB = patterns_list[0]
    patterns_AC = patterns_list[1]
    patterns_BC = patterns_list[2]
    mean_AB = means_list[0]
    mean_AC = means_list[1]
    mean_BC = means_list[2]
    std_AB = std_list[0]
    std_AC = std_list[1]
    std_BC = std_list[2]
    plotCorrelationsPerLigand(patterns_AC, tensor_binary, ligand_names_ranked, img_FilePath, results_filePath)
    patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs = writeCorrelationsToExcel(patterns_list, ["M1_AB", "M2_AC", "M3_BC"], results_filePath, protein_names_ranked, ligand_names_ranked, ['4 hrs', '8 hrs', '24 hrs', '48 hrs'])
    M1_cutoff_ce = 2.890480549 - mean_AB
    M2_cutoff_ce = 0.70486722 - mean_AC
    M3_cutoff_ce= 2.31192068 - mean_BC
    cutoffs_center_elbow_list = [M1_cutoff_ce, M2_cutoff_ce, M3_cutoff_ce]
    M1_cutoff_e = 2.890480549 - mean_AB
    M2_cutoff_e = 0.70486722 - mean_BC
    M3_cutoff_e= 2.31192068 - M3_mean
    cutoffs_elbow_list = [M1_cutoff_e, M2_cutoff_e, M3_cutoff_e]
    M1_cutoff_asymptotic = 4.137811156 - mean_AB
    M2_cutoff_asymptotic = 1.110098385 - mean_AC
    M3_cutoff_asymptotic=  2.342765763 - mean_BC
    cutoffs_asymptotic_list = [M1_cutoff_asymptotic, M2_cutoff_asymptotic, M3_cutoff_asymptotic]
    getSignificantEntities([patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs], cutoffs_elbow_list, cutoffs_center_elbow_list, cutoffs_asymptotic_list, [mean_AB, mean_AC, mean_BC], ["M1_Elbow", "M2_Elbow", "M3_Elbow"], ["M1_Center_Elbow", "M2_Center_Elbow", "M3_Center_Elbow"], ["M1_Asymptotic", "M2_Asymptotic", "M3_Asymptotic"], results_filePath)
    print("Completed rppa")
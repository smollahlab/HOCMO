import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from .utils import *
from .ncp import *
import statistics
from sktensor import dtensor



def createTensor(input_matrix, input_index_column, y_val, z_val):
    '''
    Performs data preprocessing and creates a 3 dimensional tensor based on an input incidience matrix. See (Ref 1) for information regarding incideience matrices and their format. Also prints size of tensor
    INPUTS:
    input_matrix: incidience matrix of hypergraph who's tensor is to be created
    input_index_column: this should correspond to one of your 3 values, in a standard incidience matrix, this should be the single dimensional variable
    y_val, z_val: these are the two convoluted variables

    OUTPUTS:
    incidence_matrix: Post-processed matrix
    incidence_matrix_binary: Binary version of the processed matrix. -1 if the original value was <0 and 1 if >=0
    x_names, y_names, z_names: List of names of variables derived from input matrix. (for y and z names to function, seperate the two with '_')
    tensor: tensor representing input

    EXAMPLE USAGE:
    > incidence_matrix,incidence_matrix_binary,protein_names,disease_names,gene_names,tensor = hocmo.createTensor('HOCMO_test.csv','CRs',5,5)

    EXAMPLE OUTPUT:
    > Size of the tensor: (5, 5, 5)
    '''

    ## Importing tensor incidence matrix
    ## Skipping much of the preprocessing as we have a very simplified matrix
    incidence_matrix = pd.read_csv(input_matrix)
    incidence_matrix = incidence_matrix.set_index(input_index_column)

    ## Dimensions of the tensor, concerning that these appear to be hard coded
    x = incidence_matrix.shape[0]
    y = y_val
    z = z_val

    ##turn to binary
    incidence_matrix_binary = incidence_matrix.copy(deep=True)
    incidence_matrix_binary[incidence_matrix_binary < 0] = -1
    incidence_matrix_binary[incidence_matrix_binary >= 0] = 1 


    ##make positive with abs
    incidence_matrix = np.absolute(incidence_matrix) 
    x_names = incidence_matrix.index # type: ignore
    tensor = incidence_matrix.to_numpy().reshape([x, y, z]).transpose([2, 0, 1]) # type: ignore ## why is it transposed like this? no reason given here
    y_names = pd.Index([v.split('_')[0] for v in incidence_matrix.columns.to_numpy().reshape([y,z]).transpose([1,0])[0]]) # type: ignore ##  some fancy matrix manipulation to get disease names
    z_names = pd.Index([v.split('_')[1] for v in incidence_matrix.columns.to_numpy().reshape([x,y])[1]])  # type: ignore
    print('Size of the tensor:',tensor.shape)
    return incidence_matrix,incidence_matrix_binary,x_names,y_names,z_names,tensor

def basicVisual(tensor, x_names, y_names,z_names, x_labels, y_labels, z_labels, img_filePath, img_name):
    '''
    Creates basic 3d scatter plot of tensor's data points. 
    INPUTS:
    tensor: 3d tensor to be visualized
    x_names, y_names,z_names: names of each dimension of the tensor, this is used to generate the labels
    x_labels, y_labels, z_labels: lists containing names of entities in each axis of the tensor
    img_filePath, img_name: path and name of file for image to be saved in 

    OUTPUTS:
    3d scatter plot

    EXAMPLE USAGE:
    > hocmo.basicVisual(tensor, 'CRs', 'Diseases', 'Genes', protein_names, disease_names,gene_names, './','test')
    '''

    ##transposes tensor and ensure no 0 zeroes are present for visualization
    tensor_T = np.transpose(tensor,[1,2,0])
    print('tensor size:',tensor_T.shape)
    x,y,z = tensor_T.nonzero()

    ##Creation of 3d tensor plot
    fig = plt.figure(figsize=(9,8))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    ##Creation of axes and labels
    ax.set_xticks([])
    ax.set(xticks=range(tensor_T.shape[0]), xticklabels=x_labels,
        yticks=range(tensor_T.shape[1]), yticklabels=y_labels,
        zticks=range(tensor_T.shape[2]), zticklabels=z_labels)  
    ax.set_xlabel(x_names, fontsize=18)
    ax.set_ylabel(y_names, fontsize=18)
    ax.yaxis.labelpad=15
    ax.set_zlabel(z_names, fontsize=18)
    ax.scatter(x, y, z, cmap='cm.coolwarm')
    plt.show()
    plt.savefig(os.path.join(img_filePath, img_name), format="png")
    return tensor_T

def getCoreConsistency(tensor, imageName, iters = 100, num_k = 11, start =2, top_k=20):

    '''
    Calculates core consistency, a diagnostic used in determining the optimal number of copomnent for analysis in multiway models. Simulated results 
    per number of component are then visualized via elbow plot.
    parafac/corncondia are packages in R developed by Panisson et al, we use them here via rpy2

    INPUTS:
    tensor: 3d tensor to be analyzed
    imageName: path to save generated elbow plot
    iters: number of iterations of simulation to be done. Default is 100
    num_k: Ceiling on number of components to be analyzed
    start: Starting number of components to be analyzed
    top_k: Cutoff number for number of results selected for each component tested. (selects top values)

    OUTPUTS:
    Elbow plot, optimal K number of components

    EXAMPLE USAGE:
    hocmo.getCoreConsistency(tensor, "./core_consistency_rppa.png", iters = 100, num_k = 11, start =2, top_k=20)

    EXAMPLE OUTPUT:
    Elbow plot
    Number of K vs. Core consistency
    2

    References:
    Gauvin, Laetitia, André Panisson, and Ciro Cattuto. 2014. “Detecting the Community Structure and Activity Patterns of Temporal Networks: A Non-Negative Tensor Factorization Approach.” Edited by Yamir Moreno. PLoS ONE 9 (1): e86028. https://doi.org/10.1371/journal.pone.0086028.
    Bro, Rasmus, and Henk A. L. Kiers. 2003. “A New Efficient Method for Determining the Number of Components in PARAFAC Models.” Journal of Chemometrics 17 (5): 274–86. https://doi.org/10.1002/cem.801.
    '''
    rpy2.robjects.numpy2ri.activate()
    importr('multiway')
    parafac = robjects.r["parafac"]
    corcondia = robjects.r["corcondia"]
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
        top_idx = np.argsort(ccvs)[-top_k:]
        top_ccs = ccvs[top_idx]
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
    plt.savefig(imageName, format = "png")
    plt.show()
    sorted_dictionary_by_mean = sorted(dictionary_k_scores.items(), key= lambda m: (m[1]['mean']), reverse = True)
    pairs = pd.DataFrame([xs_average,ys_average])
    # pairs = pairs.set_index(0).rename_axis(None)
#    pairs.columns = [[''],['']]
    print("Number of K vs. Core consistency")
    return sorted_dictionary_by_mean[0][0]

def factorizeNCP(tensor, components):
    '''
    Factorizes the nonlinear complementarity problem  via non-negative single variable decomposition. 
    SVD is used to generate latent factors for each layer of the tensor, which is then used along with the dimension of the tensor to perform non negative factorization.

    INPUTS:
    tensor: 3d tensor to be analyzed
    components: optimal number of components found by getCoreConsistency

    OUTPUTS:
    Elbow plot

    '''

    num_component = components
    X = dtensor(tensor) 
    init_factors = init_nnsvd(X, num_component)
    X_approx_ks = nonnegative_tensor_factorization(X, num_component, method='anls_bpp', 
                                                   init=init_factors.copy())

    C = X_approx_ks.U[0]
    A = X_approx_ks.U[1]
    B = X_approx_ks.U[2]
    print('[A,B,C]:', A.shape, B.shape, C.shape)
    return A, B, C


    
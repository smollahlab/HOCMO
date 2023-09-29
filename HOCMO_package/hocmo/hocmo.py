import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from .utils import *
import statistics
import warnings
warnings.filterwarnings("ignore")


def createTensor(input_matrix, input_index_column, y_val, z_val, sort=True):
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
    sort: boolean deciding whether you want your input columns to be sorted (lexographically) this may not work well for numerical data, but if comparing more than one input, will ensure that they are all in the same order. Set it to false if you wish to use a custom order.

    EXAMPLE USAGE:
    > incidence_matrix,incidence_matrix_binary,protein_names,disease_names,gene_names,tensor = hocmo.createTensor('HOCMO_test.csv','CRs',5,5)

    EXAMPLE OUTPUT:
    > Size of the tensor: (5, 5, 5)
    '''

    ## Importing tensor incidence matrix
    ## Skipping much of the preprocessing as we have a very simplified matrix
    incidence_matrix = pd.read_csv(input_matrix)
    incidence_matrix = incidence_matrix.set_index(input_index_column)
    if sort == True:
        incidence_matrix.sort_index(axis=1, inplace=True)
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
    tensor = incidence_matrix.to_numpy().reshape([x, y, z]).transpose([2, 0, 1]) # type: ignore ## why is it transposed like this? no reason given here ##reordering
    y_names = pd.Index([v.split('_')[0] for v in incidence_matrix.columns.to_numpy().reshape([y,z]).transpose([1,0])[0]]) # type: ignore ##  some fancy matrix manipulation to get disease names
    z_names = [v.split('_')[1] for v in incidence_matrix.columns.to_numpy()] # type: ignore
    z_names = pd.Index(list({value:"" for value in z_names}))  # type: ignore
    
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
    fig = plt.figure(figsize=(15,11))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    if len(x_labels)>20:
        x_labels = ''
    if len(y_labels)>20:
        y_labels = ''
    if len(z_labels)>20:
        z_labels = ''
    ##Creation of axes and labels
    ax.set_xticks([])
    ax.set(xticks=range(tensor_T.shape[0]), xticklabels=x_labels,
        yticks=range(tensor_T.shape[1]), yticklabels=y_labels,
        zticks=range(tensor_T.shape[2]), zticklabels=z_labels)  
    ax.set_xlabel(str(str(tensor_T.shape[0])+ ' '+ x_names), fontsize=18)
    ax.set_ylabel(str(str(tensor_T.shape[1]) + ' ' +y_names), fontsize=18)
    ax.yaxis.labelpad=15
    ax.set_zlabel(str(str(tensor_T.shape[2]) + ' ' +z_names), fontsize=18)
    ax.scatter(x, y, z, cmap='cm.coolwarm')
    plt.savefig(os.path.join(img_filePath, img_name), format="png")
    plt.show()
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

def factorize(tensor, components, method="ncp"):
    '''
    Performs non-negative tensor factorization with default version of NCP as described in Panisson et. al
    tensor: Tensor to be factored
    components: value found from getCoreConsistency
    method: two possible choices with NCP developed by Panisson et al. as the default. Can also choose to use NTF as constructed in the tensorly package
    See utils.py and function descriptions for details.

    OUTPUTS:
    A,B,C as latent factors, one axis divided into k components

    EXAMPLE USAGE:
    > A,B,C = hocmo.factorize(tensor, components, method="ncp")

    EXAMPLE OUTPUT:
    > [A,B,C]: (5, 2) (5, 2) (5, 2)

    References:
    Gauvin, Laetitia, André Panisson, and Ciro Cattuto. 2014. “Detecting the Community Structure and Activity Patterns of Temporal Networks: A Non-Negative Tensor Factorization Approach.” Edited by Yamir Moreno. PLoS ONE 9 (1): e86028. https://doi.org/10.1371/journal.pone.0086028.
    Bro, Rasmus, and Henk A. L. Kiers. 2003. “A New Efficient Method for Determining the Number of Components in PARAFAC Models.” Journal of Chemometrics 17 (5): 274–86. https://doi.org/10.1002/cem.801.
    '''
    if method == "ncp":
        return factorizeNCP(tensor, components)
    else:
        return factorizeTensorly(tensor, components)

def plotFactorMatrices(A, B, C, components, imgName_A, imgName_B, imgName_C, x_names, y_names,z_names):
    '''
    Plots component memership of each decomposed latent factor.
    INPUTS:
    A, B, C: individual latent factors to plot
    components: optimal number of components found from getCoreConsistency
    imgName_A, imgName_B, imgName_C: the names of each image
    x_names, y_names,z_names: used to label figures

    OUTPUTS:
    Figure of latent factors component membership. 3 graphs, each with number of columns equal to components

    EXAMPLE USAGE:
    > hocmo.plotFactorMatrices(A, B, C, components, './data/A_rppa.png', './data/B_rppa.png', './data/C_rppa.png', protein_names,ligand_names,condition_names)

    EXAMPLE OUTPUT:
    > Figure of latent factors component membership
    '''
    x_labels = ['',]
    for i in range(1,components+1):
        x_labels.append('comp_'+str(i))
    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.matshow(A)
    ax1.set_aspect('auto')
    ax1.set_title('A', y=-0.1)
    fig.colorbar(cax1)
    ax1.set_yticklabels(x_names.insert(0,''))
    ax1.set_xticklabels(x_labels)
    plt.savefig(imgName_A, format="png")
    ax2 = fig.add_subplot(132)
    cax2 = ax2.matshow(B)
    ax2.set_aspect('auto')
    ax2.set_title('B', y=-0.1)
    fig.colorbar(cax2)
    ax2.set_yticklabels(y_names.insert(0,''))
    ax2.set_xticklabels(x_labels)
    plt.savefig(imgName_B, format="png")
    ax3 = fig.add_subplot(133)
    cax3 = ax3.matshow(C)
    ax3.set_aspect('auto')
    ax3.set_title('C', y=-0.1)
    ax3.set_yticklabels(z_names.insert(0,''))
    ax3.set_xticklabels(x_labels)
    fig.colorbar(cax3)
    plt.savefig(imgName_C, format="png")
    plt.show()
    fig.colorbar(cax3)

def componentPredictionsForFactors(A, B, C):
    '''
    Classify inputs into components. Component group is determined based on largest membership values
    INPUTS:
    A,B,C: factorized outputs from factorize

    OUTPUTS:
    predicted component groups: component_preds_A, component_preds_B, component_preds_C.

    EXAMPLE USAGE:
    > component_preds_A, component_preds_B, component_preds_C = hocmo.componentPredictionsForFactors(A,B,C)

    EXAMPLE OUTPUT:
    > 5 inputs in total, each input belongs to a predicted component with the largest membership value:
    > [0 1 0 1 1]
    > 5 inputs in total, each input belongs to a predicted component with the largest membership value:
    > [0 1 0 0 0]
    > 5 inputs in total, each input belongs to a predicted component with the largest membership value:
    > [1 1 0 0 0]
    '''
    component_preds_A = A.argmax(axis=1) # classify inputs into respective components based on the largest membership values
    print('{} inputs in total, each input belongs to a predicted component with the largest membership value:\n\n'.format(A.shape[0]), component_preds_A)
    component_preds_B = B.argmax(axis=1) # classify inputs into respective components based on the largest membership values
    print('{} inputs in total, each input belongs to a predicted component with the largest membership value:\n\n'.format(B.shape[0]), component_preds_B)
    component_preds_C = C.argmax(axis=1) # classify inputs into respective components based on the largest membership values
    print('{} inputs in total, each input belongs to a predicted component with the largest membership value:\n\n'.format(B.shape[0]), component_preds_B)
    return component_preds_A, component_preds_B, component_preds_C

def getClusterMembershipProbabilityA(A, y_val, z_val, num_component, component_preds_A,  incidence_matrix_binary, incidence_matrix, y_label, img_title, imgName ):
    '''
    Get Cluster Membership Probability for factorized output A

    INPUTS:
    A: factorized outputs from factorize
    y_val, z_val: size of convoluted data, same as in createTensor
    num_component: Predicted number of components from getCoreConsistency
    component_preds: Output from componentPredictionsForFactors. Predicted component membership for each factorized layer of tensor
    incidence_matrix_binary, incidence_matrix: input tensors, created from createTensor
    y_label, img_title, imgName: label and name of output image.

    OUTPUTS:
    tensor_binary: processed version of input binary tensor. Tensor is re-ranked to match the new ranking of latent factor A.
    A_ranked, A_names_ranked, A_clusters: A is ranked based on strength of correlation. A_ranked is the actual latent factor variables. A_names_ranked is the labels of the latent factor A.
                                            A_clusters represents the components to which different entities in A belong

    EXAMPLE USAGE:
    > tensor_binary, A_ranked, A_names_ranked, A_clusters = hocmo.getClusterMembershipProbabilityA(A, components, component_preds_A,  incidence_matrix_binary, incidence_matrix,
                                                        'Components', 'Proteins', "A_ranked_per_cluster.png", )

    EXAMPLE OUTPUT:
    > Membership Probability figure.

    '''
        
    new_indexes = []
    clusters = []
    for k in range(num_component):
        idx = np.where(component_preds_A == k)[0]
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
    x_names = incidence_matrix.index
    A_ranked = A[new_indexes]
    A_names_ranked = x_names[new_indexes]
    tensor_binary = incidence_matrix_binary.to_numpy()[new_indexes].reshape([incidence_matrix.shape[0],y_val, z_val]).transpose([1, 0, 2]) ##hard coded?
    print(tensor_binary.shape)
    fig = plt.figure(figsize=(10,7))
# fig.subplots_adjust(hspace=0, wspace=0.3)
    ax = fig.add_subplot(111)
    aa = ax.matshow(A_ranked,cmap=plt.cm.gray_r) # type: ignore
    ax.set_aspect('auto')
    ax.set_title(img_title)
    ax.set_title('Components')
    ax.set_ylabel(y_label, size=(13))
    ax.set_xticks(range(0,num_component))
    fig.colorbar(aa)
    plt.savefig(imgName, format="png")
    return tensor_binary, A_ranked, A_names_ranked, clusters

def getClusterMembershipProbability(factor_matrix,num_component, component_preds, x_names, y_label, img_title, imgName):
    '''
    Get Cluster Membership Probability for generalized factorized output. Analgous to getClusterMembershipProbabilityA except tensor does not need to be ordered.

    INPUTS:
    factor_matrix: factorized outputs from factorize
    num_component: Predicted number of components from getCoreConsistency
    component_preds: Output from componentPredictionsForFactors. Predicted component membership for each factorized layer of tensor
    incidence_matrix_binary, incidence_matrix: input tensors, created from createTensor
    y_label, imgName: label and name of output image.

    OUTPUTS:
    Cluster Membership Probability for factorized output. Analgous to getClusterMembershipProbabilityA

    EXAMPLE USAGE:
    > B_ranked, B_names_ranked, B_clusters = hocmo.getClusterMembershipProbability(B, components, component_preds_B, gene_names, 'Components', 'Ligands', "B_ranked_per_cluster.png")

    EXAMPLE OUTPUT:
    > Same as output for getClusterMembershipProbabilityA
    '''
    # rerange the proteins in matrix A
    new_indices = []
    clusters = []
    for k in range(num_component):
        idx = np.where(component_preds == k)[0]
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
    B_names_ranked = x_names[new_indices]
    fig = plt.figure(figsize=(10,7))
# fig.subplots_adjust(hspace=0, wspace=0.3)
    #logging.debug("Plotting the factor matrix and labels.. ")
    ax = fig.add_subplot(111)
    aa = ax.matshow(factor_matrix_ranked,cmap=plt.cm.gray_r) # type: ignore
    ax.set_aspect('auto')
    ax.set_title(img_title)
    ax.set_ylabel(y_label, size=(13))
    ax.set_xticks(range(0,num_component))
    fig.colorbar(aa)
    fig.savefig(imgName, format="png")
    return factor_matrix_ranked, B_names_ranked, clusters

def plotLatentFactor(A_ranked, B_ranked, C, components, A_names_ranked, B_names_ranked, C_names, img_filePath): ##same as plot latent factor
    ## C assumed to have order
    '''
    Plots latent factors for ranked outputs from getClusterMembershipProbabilityA and getClusterMembershipProbability
    INPUTS:
    A_ranked, B_ranked, C,: ranked and factorized outputs from factorize and getClusterMembershipProbabilityA and getClusterMembershipProbability. Why is C not ranked?
    protein_names_ranked, ligand_names_ranked: Predicted number of components from getCoreConsistency
    img_filePath: file path for saved output image.

    OUTPUTS:
    Plotted Latent factors

    EXAMPLE USAGE:
    > hocmo.plotLatentFactor(A_ranked, B_ranked, C, A_names_ranked, B_names_ranked, './')

    EXAMPLE OUTPUT:
    > Composite latent factor figure
    '''
    x_labels = ['',]
    for i in range(1,components+1):
        x_labels.append('comp_'+str(i))
    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    ax1 = fig.add_subplot(131)
    plt.set_cmap('PiYG')
    norm = colors.TwoSlopeNorm(vcenter=0)
    cax1 = ax1.matshow(A_ranked,norm=norm)
    ax1.set_aspect('auto')
    ax1.set_title('$\mathbf{A}^r$', y=-0.1, fontsize=15)
    plt.rc('ytick',labelsize=12)
    fig.colorbar(cax1)
    plt.savefig(os.path.join(img_filePath, "A.png"), format="png")
# ax1.set_ylabel(str(tensor.shape[1])+' proteins', fontsize=14)
    ax1.set_yticks(range(len(A_names_ranked)))
    if len(A_names_ranked) > 15:
        loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
        ax1.yaxis.set_major_locator(loc)
        show_protein_names = A_names_ranked[range(0, len(A_names_ranked), 10)].insert(0,'')
        ax1.set_yticklabels(show_protein_names, fontsize=11)
    else:
        loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
        ax1.yaxis.set_major_locator(loc)
        ax1.set_yticklabels(A_names_ranked.insert(0,'')) ####How to make this graph pleasing (dynamically) for larger plots??
    ax1.set_xticklabels(x_labels, fontsize=12) ##number of components
    ax2 = fig.add_subplot(132)
    plt.set_cmap('PiYG')
    cax2 = ax2.matshow(B_ranked,norm =colors.TwoSlopeNorm(vcenter=0))
    ax2.set_aspect('auto')
    ax2.set_title('$\mathbf{B}^r$', y=-0.1, fontsize=15)
    fig.colorbar(cax2)
    plt.rc('ytick',labelsize=12)
    plt.savefig(os.path.join(img_filePath, "B.png"), format="png")
# ax2.set_ylabel(str(tensor.shape[2])+" ligands", size=(13))
    ax2.set_yticklabels(B_names_ranked.insert(0,''))
    ax2.set_xticklabels(x_labels, fontsize=12)
    ax3 = fig.add_subplot(133)
    cax3 = ax3.matshow(C, cmap='PiYG', norm =colors.TwoSlopeNorm(vcenter=0))
    ax3.set_aspect('auto')
    ax3.set_title('$\mathbf{C}^r$', y=-0.1, fontsize=15)
    plt.rc('ytick',labelsize=12)
# ax3.set_ylabel(tensor.shape[0], size=(13))
    ax3.set_yticklabels(C_names.insert(0,''))
    ax3.set_xticklabels(x_labels, fontsize=12) ## ##num_component
    plt.savefig(os.path.join(img_filePath, "C.png"), format="png")
    plt.show()
    fig.colorbar(cax3)

def saveFactorsToExcel(A_ranked, B_ranked, C, components, A_names_ranked, B_names_ranked, C_names, filePath, fileName):
    '''
    Save latent factors to excel file
    INPUTS:
    A_ranked, B_ranked, C,: ranked and factorized outputs from factorize and getClusterMembershipProbabilityA and getClusterMembershipProbability. Why is C not ranked?
    protein_names_ranked, ligand_names_ranked: Predicted number of components from getCoreConsistency
    img_filePath: file path for saved output image.

    OUTPUTS:
    Plotted Latent factors

    EXAMPLE USAGE:
    > hocmo.plotLatentFactor(A_ranked, B_ranked, C, A_names_ranked, B_names_ranked, './')

    EXAMPLE OUTPUT:
    > Composite latent factor figure
    '''
    x_labels = []
    for i in range(1,components+1):
        x_labels.append('comp_'+str(i))
    A_out = pd.DataFrame(data=A_ranked, index=A_names_ranked, columns=x_labels)
    B_out = pd.DataFrame(data=B_ranked, index=B_names_ranked, columns=x_labels)
    C_out = pd.DataFrame(data=C, index=C_names, columns=x_labels) ##num_component
    writer = pd.ExcelWriter(os.path.join(filePath, fileName))
    A_out.to_excel(writer, "A")
    B_out.to_excel(writer, "B")
    C_out.to_excel(writer, "C")
    writer.close()

def entitiesPerCluster(clusters, entity_names): ##Scatter plots
    '''
    Gives entities that exist in a cluster.

    INPUTs:
    clusters: Cluster membership found from getClusterMembershipProbabilityA and getClusterMembershipProbability.
    entity_names: name of entity who's cluster membership is in question

    EXAMPLE USAGE:
    > proteins = hocmo.entitiesPerCluster(A_clusters, gene_names)

    EXAMPLE OUTPUT:
    > Protein cluster 1: G2 G4 G5

    '''
    for k in range(len(clusters)):
        entities = ' '.join(entity_names[clusters[k]])
    print('Entity cluster {}: {}\n'.format(k, entities)) # type: ignore
    return entities # type: ignore

def plotForFactorMatrix(factor_matrix, clusters, labels_ranked, components, imgName_cluster_scatter, img_filePath ,img_name_prob):
    '''
    Calls plotClusters and plotProbabilityDistributionOfClusters to create cluster membership figures, as well as the probability distribution of clusters

    INPUT:
    factor_matrix: input matrix, input matrix with raw data
    clusters: output from getClusterMembershipProbability, predicted cluster membership
    labels_ranked: sorted list of labels
    imgName_cluster_scatter: Name of cluster membership scatterplot 
    img_filePath ,img_name_prob: Name of path for figure of cluster probability distribution

    EXAMPLE USAGE:
    > hocmo.plotForFactorMatrix(A_ranked, A_clusters, A_names_ranked, components,"proteins_per_cluster_scatter_plot.png", './' ,"proteins_per_cluster_probability.png")

    EXAMPLE OUTPUT:
    > Cluster figure and probability distritbution figure
    '''
    show_ids = plotClusters(factor_matrix, clusters, labels_ranked, os.path.join(img_filePath, imgName_cluster_scatter))
    plotProbabilityDistributionOfClusters(factor_matrix, show_ids, labels_ranked, components, img_filePath, img_name_prob)

##
def computeCorrelationForfactorMatrices(factor_matrix1, factor_matrix2, xlabels_plot, ylabels_plot, xlabel_pdc, ylabel_pdc,
                                        imgtitle_plot, imgtitle_pdc, imgfilePathPlot, imgfilePathPd,
                                       sheet_name, excel_path, file_name_excel):
    '''
    Wrapper function for findCorrelationMatrix, plotCorrelationMatrix, plotCorrelationMatrixGaussianDistribution,writeResultsToExcel

    INPUTS:
    factor_matrix1, factor_matrix2: Two matrices whose correlations we wish to find
    xlabels_plot, ylabels_plot, xlabel_pdc, ylabel_pdc: Labels for the correlation matrices and probability distribution functions
    imgtitle_plot, imgtitle_pdc, imgfilePathPlot, imgfilePathPd: titles and path for generated figures
    sheet_name, excel_path, file_name_excel: name and path of excel results to be saved

    OUTPUTS:
    Correlation Matrix, probability distribution function of correlation and saved correlation matrix as excel sheet.

    EXAMPLE USAGE:
    > hocmo.plotForFactorMatrix(A_ranked, A_clusters, A_names_ranked, components,"proteins_per_cluster_scatter_plot.png", './data/' ,"proteins_per_cluster_probability.png")

    EXAMPLE OUTPUT:
    > Figures and .xlsx
    '''
    patterns =  findCorrelationMatrix(factor_matrix1, factor_matrix2)
    plotCorrelationMatrix(patterns, xlabels_plot, ylabels_plot, imgtitle_plot, os.path.join(imgfilePathPlot, str(imgtitle_plot+".png")))
    mean, std, flatten_patterns, fitted_pdf = plotCorrelationMatrixGaussianDistribution(patterns, xlabel_pdc, ylabel_pdc, imgtitle_pdc, os.path.join(imgfilePathPd, str(imgtitle_pdc + ".png")))
    writeResultsToExcel(excel_path, file_name_excel, flatten_patterns, fitted_pdf, mean, std, sheet_name)
    return patterns,mean,std 
   
def getCorrelationsForAllFactors(A_ranked, B_ranked, C_ranked, A_names_ranked, B_names_ranked, C_names_ranked, excel_path, file_name_excel, imgFilePath, imgtitles_plot, imgtitle_pdc):
    '''
    Gets pairwise correlation scores and probability distribution matrices. Means and stds for??
    INPUTS:
    A_ranked, B_ranked, C: Latent factors
    A_names_ranked, B_names_ranked: sorted names of latent factors
    excel_path, file_name_excel, imgFilePath, imgtitles_plot, imgtitle_pdc: path for excel and images to be saved.

    OUTPUTS:
    Correlation plots, probability distribution figures, means and stds

    EXAMPLE USAGE:
    > patterns_list, means_list, std_list = hocmo.getCorrelationsForAllFactors(A_ranked, B_ranked, C, A_names_ranked, B_names_ranked, gene_names,'./', "Intensity_density_ncp_rppa.xlsx", './', [ "M1r", "M2r", "M3r"],
                                        ['Probability Density Function M1r', 'Probability Density Function M2r','Probability Density Function M3r'])
    EXAMPLE OUTPUT:
    > means, stds, Probability Density Function figures for ineach input
    '''
    patterns_BA, mean_BA, std_BA = computeCorrelationForfactorMatrices(A_ranked, B_ranked, B_names_ranked, A_names_ranked,"Intensity", "Density",
                                        imgtitles_plot[0], imgtitle_pdc[0], imgFilePath, imgFilePath,
                                       "M1_BA", excel_path,file_name_excel) ##
    patterns_CA, mean_CA, std_CA = computeCorrelationForfactorMatrices(A_ranked, C_ranked, C_names_ranked, A_names_ranked, "Intensity", "Density",
                                        imgtitles_plot[1], imgtitle_pdc[1], imgFilePath, imgFilePath,
                                       "M2_CA", excel_path,file_name_excel) ##
    patterns_CB, mean_CB, std_CB = computeCorrelationForfactorMatrices(B_ranked, C_ranked, C_names_ranked, B_names_ranked, "Intensity", "Density",
                                        imgtitles_plot[2], imgtitle_pdc[2], imgFilePath, imgFilePath,
                                       "M3_CB", excel_path,file_name_excel) ##
    return [patterns_BA, patterns_CA, patterns_CB], [mean_BA, mean_CA, mean_CB], [std_BA, std_CA, std_CB]
 
def plotCorrelationsPerEntity(patterns_AC, tensor_binary, A_names_ranked, B_names_ranked, C_names_ranked, img_FilePath, filePath): ##reason why its AC?
    '''
    Plots Correlation matrix and score per entity found in specific pairwise relationship.

    INPUTS:
    patterns_AC: pariwise correlation pattern
    tensor_binary: tensor_binary created from getClusterMembershipProbabilityA
    B_names_ranked: getClusterMembershipProbability
    img_FilePath, filePath: filepath to save plot

    OUTPUT:
    Correlation within pairwise relationship
    
    EXAMPLE USAGE:
    > hocmo.plotCorrelationsPerLigand(patterns_AC, tensor_binary, B_names_ranked, "./", "AC_Correlation.png")

    EXAMPLE OUTPUT:
    > means, stds, Probability Density Function figures for ineach input
    '''
    patterns_AC_by_entity = tensor_binary * patterns_AC
    norm = colors.TwoSlopeNorm(vcenter=0)
    fig = plt.figure(figsize=(16,10))
    plt.set_cmap('bwr')
    C_names_ranked = C_names_ranked.insert(0,'')
    C_names_ranked = [ str(x) for x in C_names_ranked ]
    for i in range(patterns_AC_by_entity.shape[0]):
        fig = plt.figure(figsize=(16,10))
        plt.set_cmap('bwr')
        ax = fig.add_subplot(2,3,i+1)
        aa = ax.matshow(patterns_AC_by_entity[i], norm=norm)
        ax.set_aspect('auto')
        ax.set_yticklabels(A_names_ranked.insert(0,''), size=(13))
        ax.set_xticklabels(C_names_ranked,fontsize=12)
        fig.colorbar(aa)
        ax.set_title(str(B_names_ranked[i]), y=-0.1, fontsize=16)
        plt.savefig(os.path.join(img_FilePath, str(B_names_ranked[i]+'.png')), format="png")
        plt.show()
# fitting the data with a gaussian distribution
    from scipy.stats import norm
    fig = plt.figure(figsize=(16,10))
    plt.set_cmap('bwr')
    for i in range(patterns_AC_by_entity.shape[0]):
        ax = fig.add_subplot(2,3,i+1) ##?
        flatten_AC = patterns_AC_by_entity[i].flatten()
        flatten_AC.sort()
        mean, std = norm.fit(flatten_AC)
        print('{}, mean:{}, std:{}'.format(B_names_ranked[i], mean, std))
        fitted_pdf = norm.pdf(flatten_AC, loc=mean, scale=std)
        writeResultsToExcel(filePath, "plotCorrelationsPerEntity.xlsx",flatten_AC, fitted_pdf, mean, std, "M2_AC_per_ligand") ##
        linestyle = {"linestyle":"-", "linewidth":2, "markeredgewidth":2, "elinewidth":2, "capsize":5}
        plt.errorbar(flatten_AC, fitted_pdf, color="green", **linestyle, fmt='-')
        plt.xlabel("Interaction Intensity", fontsize=13)
        plt.ylabel("Density Value", fontsize=13)
        plt.legend(['PDF (M2_'+str(B_names_ranked[i])+")"])
        plt.grid(linestyle='-.')
        plt.savefig(os.path.join(img_FilePath, 'PDF (M2_'+str(B_names_ranked[i]+').png')), format="png")
        plt.show()

def writeCorrelationsToExcel(patterns_list, sheet_names_patterns_list, filePath, A_names_ranked, B_names_ranked, C_names_ranked):
    '''
    Saves Correlation matrix to excel.
    INPUTS:
    patterns_list: list conntaining pairwise correlations
    sheet_names_patterns_list, filePath: sheet name and path of excel sheet to be saved to
    A_names_ranked, B_names_ranked, C_names_ranked: names of latent factors used as indexes for excel sheets

    OUTPUTS:
    Excel sheets containing correlation matrices, with the same data as dataframes stored in patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs

    EXAMPLE USAGE:
    > patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs = hocmo.writeCorrelationsToExcel(patterns_list, ["M1_AB", "M2_AC", "M3_BC"], './data/correlations.xlsx', A_names_ranked, B_names_ranked, gene_names)

    EXAMPLE OUTPUT:
    > pairwise correlation matrix as dataframes
    '''
    patterns_AB_pairs = pd.DataFrame(data=patterns_list[0], index=A_names_ranked, columns=B_names_ranked)
    patterns_AC_pairs = pd.DataFrame(patterns_list[1], index=A_names_ranked, 
                              columns=C_names_ranked)
    patterns_BC_pairs = pd.DataFrame(data=patterns_list[2], index=B_names_ranked, 
                      columns=C_names_ranked)
    writer = pd.ExcelWriter(filePath)
    patterns_AB_pairs.to_excel(writer, sheet_names_patterns_list[0])
    patterns_AC_pairs.to_excel(writer, sheet_names_patterns_list[1])
    patterns_BC_pairs.to_excel(writer, sheet_names_patterns_list[2])
    writer.close()
    return patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs

def getSignificantEntities(patterns_list, cutoffs_elbow_list, cutoffs_center_elbow_list, cutoffs_asymptotic_list, mean_list, sheet_names_elbow, sheet_names_center_elbow, sheet_names_asymptotic, filePath):
    '''
    Filters for significant results of correlations
    INPUTS:
    patterns_list: List of correlation matrices
    cutoffs_elbow_list, cutoffs_center_elbow_list, cutoffs_asymptotic_list, mean_list: list of cutoffs values to determine significance
    sheet_names_elbow, sheet_names_center_elbow, sheet_names_asymptotic: names of sheets to be saved in the excel sheet 
    filePath: filePath of saved excel sheet

    OUTPUTS:
    excel sheets containing filtered significant correlations

    EXAMPLE USAGE:
    > hocmo.getSignificantEntities([patterns_AB_pairs, patterns_AC_pairs, patterns_BC_pairs], cutoffs_elbow_list, cutoffs_center_elbow_list, cutoffs_asymptotic_list, [mean_AB, mean_AC, mean_BC], ["M1_Elbow", "M2_Elbow", "M3_Elbow"], ["M1_Center_Elbow", "M2_Center_Elbow", "M3_Center_Elbow"], ["M1_Asymptotic", "M2_Asymptotic", "M3_Asymptotic"], './data/SignificantEntities.xlsx')

    EXAMPLE OUTPUT:
    > Saved significant entities to  excel sheets
    '''
    getSignificantEntitiesForCenterElbow(patterns_list, cutoffs_center_elbow_list, mean_list, filePath, sheet_names_center_elbow)
    getSignificantEntitiesForElbow(patterns_list, cutoffs_elbow_list, mean_list, filePath, sheet_names_elbow)
    getSignificantEntitiesAsymptotic(patterns_list, cutoffs_asymptotic_list, mean_list, filePath, sheet_names_asymptotic)

def crossVisualize(patterns_AB1, patterns_AB, A_names_ranked, A_names_ranked1, output_filename):
    '''
    If you run the pipeline on more than one set of triplicate data, if they are a common axis, you can use this function to plot the relationship between the two triplcates.
    i.e. patterns AB and AB1 shared B as a common axis. Then A and A1 can be plotted against one another
    INPUTS:
    patterns_AB1, patterns_AB: The two outputs from two seperate tensors that share a common axis.
    A_names_ranked, A_names_ranked1: corresponding labels of each output
    output_filename: the path for the generated image to be saved

    OUTPUTS:
    correlation matrix image of latent factors from two seperate tensors that share a common axis.

    EXAMPLE USAGE:
    > patterns_rppa_gcp = hocmo.crossVisualize(patterns_AB1, patterns_AB, A_names_ranked, A_names_ranked1, "data/M4_with_Intensity_dist_ncp.xlsx")

    EXAMPLE OUTPUT:
    > Saved significant image
    '''
    patterns_AB1_expand = np.expand_dims(patterns_AB1, axis=1)
    patterns_1_2 = np.sum(np.transpose(np.multiply(patterns_AB, patterns_AB1_expand), [1, 0, 2]), axis=2)
    print(patterns_1_2.shape)

    fig = plt.figure(figsize=(5, 7))
    ax = fig.add_subplot(111)
    aa = ax.matshow(patterns_1_2, cmap='bwr', norm=colors.TwoSlopeNorm(vcenter=0))
    ax.set_aspect('auto')

    ax.set_yticks(range(len(A_names_ranked)))
    loc = plticker.MultipleLocator(base=10)
    ax.yaxis.set_major_locator(loc)
    show_protein_names = A_names_ranked[range(0, len(A_names_ranked), 10)].insert(0, '')
    ax.set_yticklabels(show_protein_names, fontsize=11)

    ax.set_xticks(range(len(A_names_ranked1)))
    loc = plticker.MultipleLocator(base=3.2)
    ax.xaxis.set_major_locator(loc)
    show_2_names = A_names_ranked1[range(0, len(A_names_ranked1), 3)].insert(0, '')
    ax.set_xticklabels(show_2_names, fontsize=11, rotation=90)

    ax.set_title("$\mathbf{M}_4$", y=-0.1, fontsize=16)
    fig.colorbar(aa)

    patterns_1_2_pairs = pd.DataFrame(data=patterns_1_2, index=A_names_ranked, columns=A_names_ranked1)

    with pd.ExcelWriter(output_filename) as writer_rppa_gcp:
        patterns_1_2_pairs.to_excel(writer_rppa_gcp, "M4")

    # Fitting the data with a gaussian distribution
    fig = plt.figure(figsize=(7, 4.3))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=14)

    patterns_1_2_ = patterns_1_2.flatten()
    patterns_1_2_.sort()
    mean, std = norm.fit(patterns_1_2_)
    print('mean:{}, std:{}'.format(mean, std))

    fitted_pdf = norm.pdf(patterns_1_2_, loc=mean, scale=std)
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 5}
    plt.errorbar(patterns_1_2_, fitted_pdf, color="green", **linestyle, fmt='-')
    plt.xlabel("Interaction Intensity", fontsize=16)
    plt.ylabel("Density Value", fontsize=16)
    plt.legend(['Probability Density Function (M4)'])
    plt.grid(linestyle='-.')
    plt.show()
    return patterns_1_2


def getAllHOCScores(patterns_1, patterns_2, names_1_ranked, names_2_ranked, names_3_ranked, filePath):
    '''
    Find the higher-order correlation score. Two pairwise correlation patterns (created from the same tensor) that have the same number of rows are multipled. 

    INPUTS:
    patterns_1, patterns_2: the two pairwse correlation patterns generated from the same tensor, and have the same number of rows
        Dimension of output files is determined by pattern 2. (wil have the same output dimensions)
    names_1_ranked, names_2_ranked,names_3_ranked: the ranked labels corresponding to each pattern
    filePath: filePath of saved excel sheet

    OUTPUTS:
    excel sheets containing filtered significant correlations

    EXAMPLE USAGE:
    > total_hoc_score = hocmo.getAllHOCScores(patterns_AB, patterns_AC, A_names_ranked, C_names_ranked, B_names_ranked,"data/HOC_scores.xlsx")

    EXAMPLE OUTPUT:
    > An .xlsx file is created with a number of sheets equal to the number of elements in names_3_ranked +2. 
    A correlation matrix is created for each entity along the common axis and saved to a sheet. (e.g. for each element in B, we create an excel sheet with elements of A as rows and elements of C as columns, the score is stored in this sheet)
    We also provide a list of all triplicates and their score. This is an alternative representation of data stored in the excel sheets.
    We also provide a sheet with some summary statistics for each element in B (i.e. each sheet in the .xlsx file)
    '''
    writer_hoc_score = pd.ExcelWriter(filePath)
    total_hoc_score = pd.DataFrame()
    rowlist = []
    for i in range(patterns_1.shape[1]):
        summ_dict = {}
        patterns_1_i = np.expand_dims(patterns_1[:, i], axis=0)
        patterns_1_2_i = patterns_2 * patterns_1_i.T
        hoc_score_i = pd.DataFrame(data=patterns_1_2_i, index=names_1_ranked, columns=names_2_ranked)
        hoc_score_i.to_excel(writer_hoc_score, names_3_ranked[i])
        entity_np = hoc_score_i.to_numpy().flatten()
        entity_np_sorted = sorted(entity_np)
        summ_dict["B_entities"] = names_3_ranked[i]
        summ_dict["min"] = np.min(entity_np)
        summ_dict["1Q"] = np.percentile(entity_np_sorted, 25)
        summ_dict["median"] = np.percentile(entity_np_sorted, 50)
        summ_dict["3Q"] = np.percentile(entity_np_sorted, 75)
        summ_dict["max"] = np.max(entity_np)
        rowlist.append(summ_dict)
        hoc_score_i_reset = hoc_score_i.reset_index()
        result_hoc_score_i = hoc_score_i_reset.melt(id_vars=['Protein'], var_name='histone', value_name='HOC Score')
        result_hoc_score_i['Triplet'] = '<' + result_hoc_score_i['Protein'] + ', ' + result_hoc_score_i['histone'] + ', ' + names_3_ranked[i] + '>'
        result_hoc_score_i = result_hoc_score_i[['Triplet', 'HOC Score']]
        total_hoc_score = pd.concat([total_hoc_score, result_hoc_score_i])
    summary_df = pd.DataFrame(rowlist)
    summary_df.to_excel(writer_hoc_score, 'Summary Statistics')
    total_hoc_score = total_hoc_score.sort_values('HOC Score', ascending=False)
    total_hoc_score.to_excel(writer_hoc_score, 'All Triplets')
    writer_hoc_score.close()
    return total_hoc_score

def getFilteredHOCScores(patterns_1, patterns_2, names_1_ranked, names_2_ranked, sig_3, sig_1, sig_2, filePath):
    '''
    Analgous to getAllHOCScores, with the added ability to run HOCMO on selected elements in each latent factor and normalize the HOC score for the filtered entities

    INPUTS:
    patterns_1, patterns_2: the two pairwse correlation patterns generated from the same tensor, and have the same number of rows
        Dimension of output files is determined by pattern 2. (wil have the same output dimensions)
    names_1_ranked, names_2_ranked,names_3_ranked: the ranked labels corresponding to each pattern
    sig_3, sig_1, sig_2: the significant entities in each pattern.
    filePath: filePath of saved excel sheet

    OUTPUTS:
    excel sheets containing filtered significant correlations

    EXAMPLE USAGE:
    > sig_ligands = B_names_ranked
    > sig_proteins = ['CCNB1','PLK1','RB1pS807S811','RPS6pS240S244','DUSP4','RPS6pS235S236',
                'CDC2','MYH2pS1943']
    > sig_histones = ['H3K9me2S10ph1K14ac0 ','H3K9me3S10ph1K14ac0 ','H3K18ub1K23ac0 ']
    > total_hoc_score = hocmo.getFilteredHOCScores(patterns_AB, patterns_rppa_gcp, A_names_ranked, A_names_ranked1, B_names_ranked,sig_proteins, sig_histones, "data/filtered_HOC_scores.xlsx")


    EXAMPLE OUTPUT:
    > An .xlsx file is created with a number of sheets equal to the number of elements in names_3_ranked +2. 
    For the filtered HOC score, all scores are normalized to range of [0,1] based on the maximum score.
    A correlation matrix is created for each entity along the common axis and saved to a sheet. (e.g. for each element in B, we create an excel sheet with elements of A as rows and elements of C as columns, the score is stored in this sheet)
    We also provide a list of all triplicates and their score. This is an alternative representation of data stored in the excel sheets.
    We also provide a sheet with some summary statistics for each element in B (i.e. each sheet in the .xlsx file)
    '''
    writer_hoc_score = pd.ExcelWriter(filePath)
    total_hoc_score = pd.DataFrame()
    rowlist = []
    for i in range(patterns_1.shape[1]):
        summ_dict = {}
        patterns_1_i = np.expand_dims(patterns_1[:, i], axis=0)
        patterns_1_2_i = patterns_2 * patterns_1_i.T
        hoc_score_i = pd.DataFrame(data=patterns_1_2_i, index=names_1_ranked, columns=names_2_ranked)
        hoc_score_i = hoc_score_i.loc[sig_1, sig_2]
        hoc_score_i = hoc_score_i/hoc_score_i.max().max()
        hoc_score_i.to_excel(writer_hoc_score, sig_3[i])
        entity_np = hoc_score_i.to_numpy().flatten()
        entity_np_sorted = sorted(entity_np)
        summ_dict["B_entities"] = sig_3[i]
        summ_dict["min"] = np.min(entity_np)
        summ_dict["1Q"] = np.percentile(entity_np_sorted, 25)
        summ_dict["median"] = np.percentile(entity_np_sorted, 50)
        summ_dict["3Q"] = np.percentile(entity_np_sorted, 75)
        summ_dict["max"] = np.max(entity_np)
        rowlist.append(summ_dict)
        hoc_score_i_reset = hoc_score_i.reset_index()
        result_hoc_score_i = hoc_score_i_reset.melt(id_vars=['Protein'], var_name='histone', value_name='HOC Score')
        result_hoc_score_i['Triplet'] = '<' + result_hoc_score_i['Protein'] + ', ' + result_hoc_score_i['histone'] + ', ' + sig_3[i] + '>'
        result_hoc_score_i = result_hoc_score_i[['Triplet', 'HOC Score']]
        total_hoc_score = pd.concat([total_hoc_score, result_hoc_score_i])
    summary_df = pd.DataFrame(rowlist)
    summary_df.to_excel(writer_hoc_score, 'Summary Statistics')
    total_hoc_score = total_hoc_score.sort_values('HOC Score', ascending=False)
    total_hoc_score.to_excel(writer_hoc_score, 'All Triplets')
    writer_hoc_score.close()
    return total_hoc_score

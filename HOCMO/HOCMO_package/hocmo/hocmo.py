import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from .utils import *
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

def factorize(tensor, components, method="ncp"):
    '''
    Performs non-negative tensor factorization with default version of NCP as described in Panisson et. al
    tensor: Tensor to be factored
    components: value found from getCoreConsistency
    method: two possible choices with NCP developed by Panisson et al. as the default. Can also choose to use NTF as constructed in the tensorly package
    See utils.py and function descriptions for details.

    OUTPUTS:
    ?

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

def plotFactorMatrices(A, B, C, tensor, imgName_A, imgName_B, imgName_C, y_names):
    '''
    What is actually plotted here? Need context so I can figure out how to rename all the axes.
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
    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.matshow(A)
    ax1.set_aspect('auto')
    ax1.set_title('A', y=-0.1)
    fig.colorbar(cax1)
    ax1.set_ylabel(str(tensor.shape[1])+' proteins', size=(13))
    ax1.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    plt.savefig(imgName_A, format="png")
    ax2 = fig.add_subplot(132)
    cax2 = ax2.matshow(B)
    ax2.set_aspect('auto')
    ax2.set_title('B', y=-0.1)
    fig.colorbar(cax2)
    ax2.set_yticklabels(y_names.insert(0,''))
    ax2.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    #plt.show()
    plt.savefig(imgName_B, format="png")
    ax3 = fig.add_subplot(133)
    cax3 = ax3.matshow(C)
    ax3.set_aspect('auto')
    ax3.set_title('C', y=-0.1)
    ax3.set_yticklabels(['','4_hour','8_hour','24_hour','48_hour'])
    ax3.set_xticklabels(['','com_1', 'comp_2', 'comp_3'])
    fig.colorbar(cax3)
    plt.show()
    plt.savefig(imgName_C, format="png")
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
    plt.savefig(imgName, format="png")
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
'''
def callRppaDataProcessingPipeline(factorization_method, filePath_rppa, fileName_rppa, filePath_sig_proteins, fileName_sig_proteins, img_FilePath, results_filePath):
    rppa_averaged, rppa_averaged_binary, tensor, protein_names, ligand_names = createRppaTensor(filePath_rppa, fileName_rppa, filePath_sig_proteins, fileName_sig_proteins)
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

'''


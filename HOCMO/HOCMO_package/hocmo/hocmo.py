import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def createTensor(input_matrix, input_index_column, y_val, z_val):
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
    incidence_matrix_binary[incidence_matrix_binary >= 0] = 1 ##no 0? only -1 or 1?


    #make positive with abs
    incidence_matrix = np.absolute(incidence_matrix) 
    protein_names = incidence_matrix.index
    tensor = incidence_matrix.to_numpy().reshape([x, y, z]).transpose([2, 0, 1]) ## why is it transposed like this? no reason given here
    disease_names = pd.Index([v.split('_')[0] for v in incidence_matrix.columns.to_numpy().reshape([y,z]).transpose([1,0])[0]]) ## just some fancy matrix manipulation to get disease names
    gene_names = pd.Index([v.split('_')[1] for v in incidence_matrix.columns.to_numpy().reshape([x,y])[1]]) 
    print('Size of the tensor:',tensor.shape)
    print(disease_names)
    return incidence_matrix,incidence_matrix_binary,protein_names,disease_names,gene_names,tensor

def basicVisual(tensor, protein_names, disease_names,gene_names):
    tensor_T = np.transpose(tensor,[1,2,0])
    print('tensor size:',tensor_T.shape)
    x,y,z = tensor_T.nonzero()
    fig = plt.figure(figsize=(9,8))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    # ax.set_yticks([0,1,2,3,4,5])
    # ax.set_zticks([0,1,2,3,4])
    ax.set(xticks=range(tensor_T.shape[0]), xticklabels=protein_names,
        yticks=range(tensor_T.shape[1]), yticklabels=disease_names,
    #        xticks=range(tensor_T.shape[0]), xticklabels=protein_names)
        zticks=range(tensor_T.shape[2]), zticklabels=gene_names)  
    ax.set_xlabel('CRs', fontsize=18)
    ax.set_ylabel('Diseases', fontsize=18)
    ax.yaxis.labelpad=15
    ax.set_zlabel('Genes', fontsize=18)
    c = (z<=0)
    ax.scatter(x, y, z, cmap='cm.coolwarm')
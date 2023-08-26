import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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
    tensor: tensor representing hypergraph of input

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
    incidence_matrix_binary[incidence_matrix_binary >= 0] = 1 ##no 0? only -1 or 1?


    ##make positive with abs
    incidence_matrix = np.absolute(incidence_matrix) 
    x_names = incidence_matrix.index
    tensor = incidence_matrix.to_numpy().reshape([x, y, z]).transpose([2, 0, 1]) ## why is it transposed like this? no reason given here
    y_names = pd.Index([v.split('_')[0] for v in incidence_matrix.columns.to_numpy().reshape([y,z]).transpose([1,0])[0]]) ##  some fancy matrix manipulation to get disease names
    z_names = pd.Index([v.split('_')[1] for v in incidence_matrix.columns.to_numpy().reshape([x,y])[1]]) 
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
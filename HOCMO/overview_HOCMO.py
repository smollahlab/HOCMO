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

    '''gcppbs_file = os.path.join(filePath_gcp,fileName_gcp)
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
    '''
def coreconsistency():
    '''def compute_namda(A, B, C):
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
    '''

def factorizeNTF():
    '''def factorizeNCP(tensor, components):
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
            return factorizeTensorly(tensor, components)'''

def visualization():
    '''def plotLatentFactor(A_ranked, B_ranked, C, protein_names_ranked, ligand_names_ranked, img_filePath):
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
        fig.colorbar(cax3)'''

def calculateCorrelationMatrices():
    '''#logging.debug("Creating a correlation matrix for the given factor matrices.")
    B_expand = np.expand_dims(factor_matrix2, axis=1)
    patterns = np.sum(np.transpose(np.multiply(factor_matrix1, B_expand), [1,0,2]), axis=2)
    print(patterns.shape)
    return patterns '''

def calculateHOCScores(correlationMatrix1, correlationMatrix2):
     '''for i in range(patterns_AB.shape[1]):
        ligand_i = np.expand_dims(patterns_AB[:,i], axis=0)
        patterns_rppa_gcp_i = patterns_rppa_gcp * ligand_i.T
        hoc_score_i = pd.DataFrame(data=patterns_rppa_gcp_i, index= protein_names_ranked, columns=histone_names_ranked)
#     hoc_score_i.to_excel(writer_hoc_score, ligand_names_ranked[i])
        filtered_hoc_score_i = hoc_score_i.loc[sig_proteins, sig_histones]
        filtered_hoc_score_i = filtered_hoc_score_i/filtered_hoc_score_i.max().max()
        filtered_hoc_score_i.to_excel(filtered_writer_hoc_score, ligand_names_ranked[i])
        ## Create a dictionary for HOC tuples -> sort in descending order '''

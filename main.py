import factorization

if __name__ == '__main__':
    filePath_rppa = "Z:/reetika/TME/dataset/rppa/"
    filePath_sig_proteins =  "Z:/reetika/TME/dataset/tumour_microenv_results_2020-9/rppa/"
    img_FilePath_rppa = "Z:/reetika/TME/result_figures/rppa/" 
    results_filePath_rppa = "Z:/reetika/TME/results_files/rppa/"
    factorization.callRppaDataProcessingPipeline("ncp", filePath_rppa, 'MDD_RPPA_Level3_preprocessed_2020-9.xlsx', filePath_sig_proteins, 'pval_005_rppa.xlsx', img_FilePath_rppa, results_filePath_rppa)
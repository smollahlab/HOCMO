# Chromatin Regulator Ontology (CRO)

An ontology to describe objects in the chromatin regulator domain. Developed using [Protege 5.6.1](http://protegeproject.github.io/protege/installation/).

## Ontology Structure
CRO defines a native taxonomy that all chromatin regulators can be classified into. CRO imports [GO v2023-06-11 (123 MB)](http://purl.obolibrary.org/obo/go/releases/2023-06-11/go.owl) and  [DOID v2023-05-31 (29.6 MB)](http://purl.obolibrary.org/obo/doid/releases/2023-05-31/doid.owl) to provide terms relating to molecular functiones, biological processes, cellular components, and disease characteristics. There are three additional top level classes defined CRO:
- `protein` represents proteins that are affected downstream by CRs but are not themselves CRs.
- `domain` represents protein domains.
-  `HistoneModifications` is further separated into `H2AModification`, `H2BModification`, `H3Modification`, and `H4Mofication` to represent covalent histone modifications such as H3K9me1. 

Most properties are imported as part of  [GO](http://purl.obolibrary.org/obo/go/releases/2023-06-11/go.owl) and [DOID](http://purl.obolibrary.org/obo/doid/releases/2023-05-31/doid.owl) imports. Many properties are also natively defined in CRO based on their frequency of appearance in CR-related journal articles. 

Chromatin regulator instances along with their respective complexes, domains, diseases, protein-protein interactions, and GO terms are encoded in CRO as proof of concept. The main example, involving the relationship between BRPF1 and heptacellular carcinoma comes from [this journal publication](https://doi.org/10.1038/s42003-021-02405-6). Additionally, 15 chromatin regulators chosen from [crewDB tables](https://github.com/smollahlab/crewATLAS/tree/master/NLP) were incorporated. Information that was not found in the tables was based on research from [UniProt](https://www.uniprot.org/), [HGNC](https://www.genenames.org/), [InterPro](https://www.ebi.ac.uk/interpro/entry/InterPro/#table), and [StringDB](https://string-db.org/). 

## References
#### Ontology Design
- Arp R, Smith B, Spear AD. Building ontologies with basic formal ontology. Cambridge, MA: The MIT Press; 2015.
- Cheng, C.LH., Tsang, F.HC., Wei, L. et al. Bromodomain-containing protein BRPF1 is a therapeutic target for liver cancer. Commun Biol 4, 888 (2021). https://doi.org/10.1038/s42003-021-02405-6
- Debellis, Michael. (2021). A Practical Guide to Building OWL Ontologies Using Protégé 5.5 and Plugins. 
- Noy, N. & Mcguinness, Deborah. (2001). Ontology Development 101: A Guide to Creating Your First Ontology. Knowledge Systems Laboratory. 32. 
#### Chromatin Regulator Review
- D'Oto A, Tian QW, Davidoff AM, Yang J. Histone demethylases and their roles in cancer epigenetics. _J Med Oncol Ther_. 2016;1(2):34-40.
- Flaus A, Martin DM, Barton GJ, Owen-Hughes T. Identification of multiple distinct Snf2 subfamilies with conserved structural motifs. _Nucleic Acids Res_. 2006;34(10):2887-2905. Published 2006 May 31. doi:10.1093/nar/gkl295
- Gillette TG, Hill JA. Readers, writers, and erasers: chromatin as the whiteboard of heart disease. _Circ Res_. 2015;116(7):1245-1253. doi:10.1161/CIRCRESAHA.116.303630
- Han P, Hang CT, Yang J, Chang CP. Chromatin remodeling in cardiovascular development and physiology. _Circ Res_. 2011;108(3):378-396. doi:10.1161/CIRCRESAHA.110.224287
- Lee, K., Workman, J. Histone acetyltransferase complexes: one size doesn't fit all. _Nat Rev Mol Cell Biol_ **8**, 284–295 (2007). https://doi.org/10.1038/nrm2145
- Marfella CG, Imbalzano AN. The Chd family of chromatin remodelers. _Mutat Res_. 2007;618(1-2):30-40. doi:10.1016/j.mrfmmm.2006.07.012
- Marmorstein R, Zhou MM. Writers and readers of histone acetylation: structure, mechanism, and inhibition. _Cold Spring Harb Perspect Biol_. 2014;6(7):a018762. Published 2014 Jul 1. doi:10.1101/cshperspect.a018762
- Mazina MY, Vorobyeva NE. Chromatin Modifiers in Transcriptional Regulation: New Findings and Prospects. _Acta Naturae_. 2021;13(1):16-30. doi:10.32607/actanaturae.11101
- Mittal, P., Roberts, C.W.M. The SWI/SNF complex in cancer — biology, biomarkers and therapy. _Nat Rev Clin Oncol_ **17**, 435–448 (2020). https://doi.org/10.1038/s41571-020-0357-3
- Morgan, M.A.J., Shilatifard, A. Reevaluating the roles of histone-modifying enzymes and their associated chromatin modifications in transcriptional regulation. _Nat Genet_ **52**, 1271–1281 (2020). https://doi.org/10.1038/s41588-020-00736-4
- Nair SS, Kumar R. Chromatin remodeling in cancer: a gateway to regulate gene transcription. _Mol Oncol_. 2012;6(6):611-619. doi:10.1016/j.molonc.2012.09.005
- Thomas T, Voss AK. The diverse biological roles of MYST histone acetyltransferase family proteins. _Cell Cycle_. 2007;6(6):696-704. doi:10.4161/cc.6.6.4013
- Zhang P, Torres K, Liu X, Liu CG, Pollock RE. An Overview of Chromatin-Regulating Proteins in Cells. _Curr Protein Pept Sci_. 2016;17(5):401-410. doi:10.2174/1389203717666160122120310
- Zhao, Z., Shilatifard, A. Epigenetic modifications of histones in cancer. _Genome Biol_ **20**, 245 (2019). https://doi.org/10.1186/s13059-019-1870-5
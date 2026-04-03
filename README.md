# Widespread deep-sea microorganisms in subseafloor geochemical cycling
#### R.L. Hageman, S. le Moine-Bauer, I. H. Thorseth, J. Brendryen, T. E. Møller,  R. Böröcz, H. R. Babel, B. Hannisdal, and S.L. Jørgensen

This repository contains the scripts and files needed to reproduce the plots provided in the article [Link soon available].

The scripts were run in the order of the file numbering in each folder, starting with 1)Preprocessing/1)FastQ_to_OTUtable continued by 1)Preprocessing/2)OTUtable_finalization/ etc.

## Data
- Sequencing data of all the studies used are in: PRJNA784957, PRJEB102573, PRJNA529480, PRJNA789780, PRJNA590088, PRJNA473406, PRJNA473406, PRJNA489438, PRJDB8106, PRJEB33873, PRJNA606622.
  - The processed data of all the studies are available in the folder 1)Preprocessing/2)OTUtable_finalization/. The OTU tables are sorted based on the abbreviations used in this study.
- The chemical data of all studies combined can be found in 1)Preprocessing/4)Significant_gene/All_MICROB_GEO_data.csv. All the data has the unit millimolar (mM), where the depth is in cm.
- The chemical data of this study is available on PANGAEA:  doi: 10.1594/PANGAEA.990258 and https://doi.pangaea.de/10.1594/PANGAEA.990499.

## Article Figure scripts
- Figure 3: ./2)Plots/dbRDA_selected_OTUs.R
- Figure 4: ./2)Plots/GC10_age_groups_comparison_geochemplot.py
- Figure 5: ./2)Plots/piechart_perSite.py

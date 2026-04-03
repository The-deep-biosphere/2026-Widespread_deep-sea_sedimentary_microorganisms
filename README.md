# [Widespread deep-sea microorganisms in subseafloor geochemical cycling](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2026.1790163/abstract)
#### R.L. Hageman, S. le Moine-Bauer, I. H. Thorseth, J. Brendryen, T. E. Møller,  R. Böröcz, H. R. Babel, B. Hannisdal, and S.L. Jørgensen

This repository contains the scripts and files needed to reproduce the plots provided in the [article](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2026.1790163/abstract).

The scripts were run in the order of the file numbering in each folder, starting with 1)Preprocessing/1)FastQ_to_OTUtable continued by 1)Preprocessing/2)OTUtable_finalization/ etc.

## Data
- Sequencing data of all the studies (be aware to cite the other studies the data was published by) used are in: PRJNA784957, PRJEB102573, [PRJNA529480](https://www.pnas.org/doi/abs/10.1073/pnas.2005917117), [PRJNA789780](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.804575/full), [PRJNA590088, PRJNA473406](https://academic.oup.com/femsec/article/96/12/fiaa223/5956488), [PRJNA489438](https://www.nature.com/articles/s41598-019-44585-6), [PRJDB8106](https://academic.oup.com/ismej/article/14/3/740/7474793?guestAccessKey=#548148848), [PRJEB33873](https://academic.oup.com/ismej/article/15/12/3455/7474371?guestAccessKey=), [PRJNA606622](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.15937).
  - The processed data of all the studies are available in the folder 1)Preprocessing/2)OTUtable_finalization/. The OTU tables are sorted based on the abbreviations used in this study.
- The chemical data of all studies combined can be found in 1)Preprocessing/4)Significant_gene/All_MICROB_GEO_data.csv. All the data has the unit millimolar (mM), where the depth is in cm.
- The chemical data of this study is available on PANGAEA: [chemical data](https://doi.pangaea.de/10.1594/PANGAEA.990258) and [14C data](https://doi.pangaea.de/10.1594/PANGAEA.990499).

## Article Figure scripts
- Figure 3: ./2)Plots/dbRDA_selected_OTUs.R
- Figure 4: ./2)Plots/GC10_age_groups_comparison_geochemplot.py
- Figure 5: ./2)Plots/piechart_perSite.py

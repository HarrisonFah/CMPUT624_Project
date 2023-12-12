To run any of the model first:

1. Download the data (original) from the following link and put the unzipped folder doi_10_5061_dryad_gt413__v20150225 in the main directory: https://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/

2. To preprocess the fMRIs, run preprocessing2.ipynb in Jupyter Notebook and run all cells


To run fMRI-CLIP code:

1. Download ResNet model from the following link and put it in in the main directory:
https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt

2. To reproduce results of the three experiments, first make sure all packages/modules are install, then optionally edit NUM_WORDS and OVERLAPPING in each of the files fMRI-CLIP/clip_loo_runs_no_dataloader.py, fMRI-CLIP/clip_loo_subjects_no_dataloader.py, and fMRI-CLIP/clip_loo_runs_subjects_no_dataloader.py and run them from the main directory

3. To reproduce line plot figures, edit values in fMRI-CLIP/line_plots.py and run it

4. To reproduce UMAP plot, run clip_loo_subjects_umap.py


To run ridge regression:

1. Download Common Crawl Corpus from https://github.com/stanfordnlp/GloVe to run GloVe

2. Run all cells in ridge_within_sub.ipynb and change the text_embedding name for the the text encoder you want to run 

3. Run all cells in ridge_LOSO.ipynb and change the text_embedding name for the the text encoder you want to run.

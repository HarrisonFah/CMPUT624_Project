To run fMRI-CLIP code:

1. Download the data (original) from the following link and put the unzipped folder doi_10_5061_dryad_gt413__v20150225 in the main directory: https://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/

2. Download ResNet model from the following link and put it in in the main directory:
https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt

3. To preprocess the fMRIs, run preprocessing2.ipynb in Jupyter Notebook and run all cells up to and including the 2nd cell under the smoothing header

4. To reproduce results of the three experiments, edit NUM_WORDS and OVERLAPPING in each of the files fMRI-CLIP/clip_loo_runs_no_dataloader.py, fMRI-CLIP/clip_loo_subjects_no_dataloader.py, and fMRI-CLIP/clip_loo_runs_subjects_no_dataloader.py and run them from the main directory

5. To reproduce line plot figures, edit values in fMRI-CLIP/line_plots.py and run it

6. To reproduce UMAP plot, run clip_loo_subjects_umap.py
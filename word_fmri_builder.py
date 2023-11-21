import transformers
from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPImageProcessor, CLIPTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
import math
import scipy.io as sio
from scipy import signal
import nibabel as nib
from pathlib import Path
from gensim.models import Word2Vec
import re
import pickle
import pandas as pd
import gzip
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

NUM_SUBJS = 8
subjects_fmri = [] #stores all 8 subject fmri np arrays

fMRI_folder = Path('./doi_10.5061_dryad.gt413__v1')
assert fMRI_folder.exists(), f"Foldder: {fMRI_folder} does not exist."

with open(fMRI_folder / 'fmri_indices', 'rb') as f:
    fmri_indices = pickle.load(f)

for subj_id in range(8):
    print("Subject:",subj_id)
    fmri_file_name = str(subj_id) + '_smooth_detrend_nifti_4d.nii'
    fmri = nib.load(fMRI_folder / fmri_file_name)
    fmri = np.array(fmri.dataobj)
    assert isinstance(fmri, np.ndarray), f"Imported fmri_scan for subject {subj_id} is not of type numpy.ndarray"
    assert(fmri.ndim) == 4, f"Imported fmri_scan for subject {subj_id} is not 4 dimensional"
    subjects_fmri.append(fmri)

feature_matrix = np.zeros((5176,195)) #stores the feature vectors as a row for each word
feature_names = [] #stores the names of all features in order
feature_types = {} #stores the types of features and all the names of the features for each type

features = sio.loadmat(fMRI_folder / 'story_features.mat')
feature_count = 0
for feature_type in features['features'][0]:
    feature_types[feature_type[0][0]] = []
    if isinstance(feature_type[1][0], str):
        feature_types[feature_type[0][0]].append(feature_type[1][0])
        feature_names.append(feature_type[1][0])
    else:
        for feature in feature_type[1][0]:
            feature_types[feature_type[0][0]].append(feature[0])
            feature_names.append(feature[0])
    feature_matrix[:, feature_count:feature_count+feature_type[2].shape[1]] = feature_type[2] #adds the (5176xN) feature values to the feature matrix for the current feature group
    feature_count += feature_type[2].shape[1]

words_info = [] #stores tuples of (word, time, features) sorted by time appeared

mat_file = fMRI_folder / 'subject_1.mat' #only looks at the first subject file, somewhere it said all the timings were the same so this should be safe
mat_contents = sio.loadmat(mat_file)
for count, row in enumerate(mat_contents['words'][0]):
    word_value = row[0][0][0][0]
    time = row[1][0][0]
    word_tuple = (word_value, time, feature_matrix[count,:])
    words_info.append(word_tuple)

#for each word, get the next 4 fMRI scans weighted by the gaussian window for each subject
#then save the word fMRI scans for each subject and the words in an pandas file
window = signal.windows.gaussian(16, std=1) #gaussian window for the 4 fMRI scans
subject_words_dict = [{'file_name':[], 'word':[], 'time':[]} for i in range(8)]
for word_count in words_info:
    word = word_count[0]
    time = word_count[1]
    print(word, time)
    fmri_count = 0
    subject_scans = []
    for i in range(1,17):
        delay = 0.5*i #time after word was read
        try:
            curr_fmri_idx = fmri_indices.index((time + delay)/2) #checks if an fMRI scan happens at this time point
            weight = window[int(2*delay)-1]
            for count, subject in enumerate(subjects_fmri):
                if fmri_count == 0:
                    subject_scans.append(weight*subject[:,:,:,curr_fmri_idx])
                else:
                    subject_scans[count] += weight*subject[:,:,:,curr_fmri_idx]
            fmri_count += 1
        except Exception as e:
            #print(e)
            pass
    print(fmri_count)
    if fmri_count == 4:
        for count, subject in enumerate(subjects_fmri):
            #save filename with (word, time) in file
            file_name = "./word_fmris/" + str(count) + "_subject_word_weighted_" + str(time) + ".pt"
            scan = torch.tensor(subject_scans[count])
            with open(file_name, 'wb') as f:
                torch.save(scan, f)
            subject_words_dict[count]['file_name'].append(file_name)
            subject_words_dict[count]['word'].append(word)
            subject_words_dict[count]['time'].append(time)
    for count, subject in enumerate(subjects_fmri):
        df = pd.DataFrame(subject_words_dict[count])
        df.to_csv("./" + str(count) + "_subject_word_fmri_labels.csv", index=False)
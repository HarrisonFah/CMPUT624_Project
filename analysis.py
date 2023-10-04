# This is how I read in the .mat files
import scipy.io as sio
mat_contents = sio.loadmat('story_features.mat')
print(mat_contents)

# This is how I read in the .npy files
import numpy as np
words = np.load('time_words_fmri.npy')
print(words)
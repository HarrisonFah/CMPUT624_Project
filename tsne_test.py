from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

num_subjs = 8
colors = ['bo-', 'go-', 'ro-', 'co-', 'mo-', 'yo-', 'ko-', 'wo-']

samples = np.random.rand(num_subjs*20, 1024)
print("samples.shape:", samples.shape)
tsne = TSNE(n_components=2, verbose=1)
tsne_samples = tsne.fit_transform(samples)
print("tsne_samples.shape:", tsne_samples.shape)

for subj in range(num_subjs):
    for i in range(10):
        plt.plot(tsne_samples[subj*10*2 + i*2: subj*10*2 + i*2 + 2, 0], tsne_samples[subj*10*2 + i*2: subj*10*2 + i*2 + 2, 1], colors[subj])

plt.show()


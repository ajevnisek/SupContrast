import torch
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tqdm import tqdm

from collections import namedtuple

from main_supcon import set_loader
from networks.resnet_big import SupConResNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home/uriel/dev/SupContrast/save/SupCon/path_models/SimCLR_path_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/last.pth'

model = SupConResNet(name='resnet50')
model.load_state_dict(torch.load(model_path)['model'])
model = model.to(device)

Opt = namedtuple('Opt', ['dataset', 'data_folder', 'mean', 'std',
                         'batch_size', 'num_workers', 'size'])
opt = Opt(dataset='path',
          data_folder='/home/uriel/dev/SupContrast/dataset/Deepfakes'
                      '/train_subset',
          mean="(0.4914, 0.4822, 0.4465)",
          std="(0.2675, 0.2565, 0.2761)",
          batch_size=16, num_workers=8,
          size=32)
train_loader = set_loader(opt)

import matplotlib.pyplot as plt
# for i in range(128):
#     plt.subplot(16, 8, i+1)
#     plt.imshow(x[0][0][i].permute(1,2,0))
#     plt.xticks([]); plt.yticks([])
# plt.show()
#
# for i in range(128):
#     plt.subplot(16, 8, i+1)
#     plt.imshow(x[0][1][i].permute(1,2,0))
#     plt.xticks([]); plt.yticks([])
# plt.show()

manipulated_features = []
original_features = []

for idx, (images, labels) in tqdm(enumerate(train_loader)):
    images = torch.cat([images[0], images[1]], dim=0)
    images = images.to(device)
    labels = labels.to(device)
    bsz = labels.shape[0]

    # compute loss
    features = model(images)
    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    # features.shape = [batch_size, views, feature_dim]
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    temp = features[labels == 0]
    batch_size = temp.shape[0]
    views = temp.shape[1]
    manipulated_features.append(
        temp.reshape(batch_size * views, temp.shape[-1]).cpu().detach())
    temp = features[labels == 1]
    batch_size = temp.shape[0]
    views = temp.shape[1]
    original_features.append(temp.reshape(batch_size * views, temp.shape[
        -1]).cpu().detach())


from tqdm import tqdm
import torch

from collections import namedtuple

from main_supcon import set_loader
from networks.resnet_big import SupConResNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home/uriel/dev/SupContrast/save/SupCon/path_models/SimCLR_path_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/last.pth'

model = SupConResNet(name='resnet50')
model.load_state_dict(torch.load(model_path)['model'])
model = model.to(device)

Opt = namedtuple('Opt', ['dataset', 'data_folder', 'mean', 'std',
                         'batch_size', 'num_workers', 'size'])
opt = Opt(dataset='path',
          data_folder='/home/uriel/dev/SupContrast/dataset/Deepfakes'
                      '/train_subset',
          mean="(0.4914, 0.4822, 0.4465)",
          std="(0.2675, 0.2565, 0.2761)",
          batch_size=16, num_workers=8,
          size=32)
train_loader = set_loader(opt)

import matplotlib.pyplot as plt
# for i in range(128):
#     plt.subplot(16, 8, i+1)
#     plt.imshow(x[0][0][i].permute(1,2,0))
#     plt.xticks([]); plt.yticks([])
# plt.show()
#
# for i in range(128):
#     plt.subplot(16, 8, i+1)
#     plt.imshow(x[0][1][i].permute(1,2,0))
#     plt.xticks([]); plt.yticks([])
# plt.show()

manipulated_features = []
original_features = []

for idx, (images, labels) in tqdm(enumerate(train_loader)):
    images = torch.cat([images[0], images[1]], dim=0)
    images = images.to(device)
    labels = labels.to(device)
    bsz = labels.shape[0]

    # compute loss
    features = model(images)
    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    # features.shape = [batch_size, views, feature_dim]
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    temp = features[labels == 0]
    batch_size = temp.shape[0]
    views = temp.shape[1]
    manipulated_features.append(
        temp.reshape(batch_size * views, temp.shape[-1]).cpu().detach())
    temp = features[labels == 1]
    batch_size = temp.shape[0]
    views = temp.shape[1]
    original_features.append(temp.reshape(batch_size * views, temp.shape[
        -1]).cpu().detach())

original_features = torch.cat(original_features, axis=0)
manipulated_features = torch.cat(manipulated_features, axis=0)

from sklearn.manifold import TSNE
all_features = torch.cat([original_features, manipulated_features], axis=0)
all_labels = torch.cat([torch.ones(original_features.shape[0], 1),
                        torch.zeros(manipulated_features.shape[0], 1)])
X = all_features.numpy()


X = all_features.numpy()
feat_cols = ['feature'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X,columns=feat_cols)
all_labels = torch.cat([torch.zeros(original_features.shape[0], 1),
                        torch.ones(manipulated_features.shape[0], 1)])
df['y'] = all_labels
df['label'] = df['y'].apply(lambda i: str(i))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.scatter(pca_result[:,0], pca_result[:,1])
plt.show()
import numpy as np
rndperm = np.random.permutation(df.shape[0])

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1]
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.scatter(tsne_results[:,0], tsne_results[:,1])

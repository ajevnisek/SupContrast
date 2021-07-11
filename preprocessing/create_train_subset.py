import os
import random


src_root = '/home/uriel/dev/SupContrast/dataset/Deepfakes/train/manipulated/'
dst_root = '/home/uriel/dev/SupContrast/dataset/Deepfakes/train_subset/manipulated/'
l = os.listdir(src_root)
to_copy = random.sample(list(set(l)), k=int(0.1 * len(l)))
for x in to_copy:
    if not os.path.exists(os.path.join(dst_root, x)):
        os.symlink(os.path.join(src_root, x), os.path.join(dst_root, x))


src_root = '/home/uriel/dev/SupContrast/dataset/Deepfakes/train/original/'
dst_root = '/home/uriel/dev/SupContrast/dataset/Deepfakes/train_subset/original/'
l = os.listdir(src_root)
to_copy = random.sample(list(set(l)), k=int(0.1 * len(l)))
for x in to_copy:
    if not os.path.exists(os.path.join(dst_root, x)):
        os.symlink(os.path.join(src_root, x), os.path.join(dst_root, x))

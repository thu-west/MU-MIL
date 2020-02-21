import os

import numpy as np
import torch
import torch.utils.data as data_utils
import imgaug as ia
from imgaug import augmenters as iaa
import scipy.io as sio
from scipy.stats import ortho_group
import imageio

dataset_path = "../datasets/DeliciousMIL/Data"

class DMILBag:
    def __init__(self):
        self.filename = os.path.join(dataset_path, name2file[name])
        self.all_bags = self.load_bags()

        # np.random.seed(10)
        shuffled_ids = np.random.permutation(len(self.all_bags))

        devide = int(len(self.all_bags) * 0.7)
        self.train_id = shuffled_ids[:devide]
        self.test_id = shuffled_ids[devide + 1:]

    def load_bags(self):
        dataset_content = sio.loadmat(self.filename)
        bag_ids = dataset_content["bag_ids"][0]  # N
        features = np.asarray(dataset_content["features"].todense())  # N x K
        labels = np.asarray(dataset_content["labels"].todense())[0]  # N

        # print(bag_ids.shape)
        # print(labels.shape)

        all_bags = []

        for i_bag in range(1, max(bag_ids) + 1):
            inst_indicator = bag_ids == i_bag
            data = features[inst_indicator, :]
            inst_labels = np.asarray(labels[inst_indicator] > 0, dtype=np.int)
            bag_label = max(inst_labels)
            all_bags.append((
                data, [bag_label, inst_labels]
            ))

        return all_bags

    def get_train(self):
        ret = []
        for trainid in self.train_id:
            ret.append(
                self.all_bags[trainid]
            )
        return ret

    def get_test(self):
        ret = []
        for testid in self.test_id:
            ret.append(
                self.all_bags[testid]
            )
        return ret


if __name__ == "__main__":
    cb = ClassicBag()

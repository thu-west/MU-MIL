import os
import numpy as np
import torch
import torch.utils.data as data_utils
import imgaug as ia
from imgaug import augmenters as iaa
import scipy.io as sio
from scipy.stats import ortho_group
import imageio

img_dir = "../datasets/BreastCancer/imgs"
mask_dir = "../datasets/BreastCancer/seg"


seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),
    # iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    # iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=[0, 90, 180, 270],
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


HCD = np.array([
    [1.88, -0.07, -0.60],
    [-1.02, 1.13, -0.48],
    [-0.55, -0.13, 1.57]
])


def get_img_mask(img_path, mask_path):
    img = np.asarray(imageio.imread(img_path))
    mask = imageio.imread(mask_path)
    sub_imgs = []
    sub_labels = []
    for ix in range(24):
        for iy in range(28):
            xmin = ix * 32
            xmax = xmin + 32
            ymin = iy * 32
            ymax = ymin + 32
            sub_fig = img[xmin: xmax, ymin: ymax, :]
            if np.sum(sub_fig == 0) > 32 * 32 * 0.75:
                continue

            assert sub_fig.shape == (32, 32, 3)

            sub_fig = sub_fig.reshape(32, 32, 1, 3)
            hcpatch = np.sum(np.multiply(sub_fig, HCD), -1)
            hcpatch = hcpatch.reshape(1, 32, 32, 3).astype(np.uint8)
            sub_imgs.append(hcpatch)

            sub_mask = np.asarray(mask[xmin: xmax, ymin: ymax])
            sub_labels.append(int(np.sum(sub_mask > 0) > 10))
    stack_imgs = np.concatenate(sub_imgs, 0)

    return stack_imgs, sub_labels


class BCBags(data_utils.Dataset):
    def __init__(self, seed=10, aug_times=5):
        self.train = True

        self.labels = []
        self.patches = []
        self.eids = []
        self.o_img = []
        self.o_labels = []
        self.o_patches = []
        self.o_eids = []
        self.o_xys = []

        self.o_labels = []
        self.o_patches = []
        self.o_eids = []

        for musk_file in os.listdir(mask_dir):
            img_file = musk_file.split(".")[0] + "_ccd.tif"
            bag_label = int("benign" in img_file)
            mask_path = os.path.join(mask_dir, musk_file)
            img_path = os.path.join(img_dir, img_file)
            ins_imgs, ins_labels = get_img_mask(img_path=img_path, mask_path=mask_path)
            self.o_labels.append(bag_label)
            self.o_patches.append(ins_imgs)
            self.o_eids.append(ins_labels)


        np.random.seed(seed)
        shuffled_ids = np.random.permutation(len(self.o_labels))
        devide = int(len(self.o_labels)*0.7)
        self.train_id = shuffled_ids[:devide]
        self.test_id = shuffled_ids[devide:]

        for _ in range(aug_times):
            a_labels, a_eids, a_patches = self.data_argument()
            self.labels += a_labels
            self.eids += a_eids
            self.patches += a_patches

        a_labels, a_eids, a_patches = self.data_argument(False)
        self.labels += a_labels
        self.eids += a_eids
        self.patches += a_patches
        np.random.seed(10)
        shuffled_ids = np.random.permutation(len(self.labels))

        devide = int(len(self.labels) * 0.7)
        self.train_id = shuffled_ids[:devide]
        self.test_id = shuffled_ids[devide:]

    def data_argument(self, train=True, Trans=True):
            r_labels, r_eids, r_patches = [], [], []

            for trainid in range(len(self.o_labels)):
                target_patches = self.o_patches[trainid]
                # if np.random.uniform(0, 1, 1) > 0.5:
                #     return target_patches
                # else:
                #     rand_scale = np.random.normal(1, 0.1, target_patches.shape)
                #     tmp_patches = np.multiply(target_patches, rand_scale)
                #     image = tmp_patches / np.max(tmp_patches, axis=(2, 3), keepdims=True) * 256
                #     return image
                if Trans:
                    target_patches = self.argue_op(target_patches)
                r_labels.append(self.o_labels[trainid])
                r_eids.append(self.o_eids[trainid])
                r_patches.append(target_patches)
            return r_labels, r_eids, r_patches

    def argue_op(self, target_patches, Trans=True):
        if Trans:
            rand_scale = np.random.normal(1, 0.1, target_patches.shape)
            tmp_patches = np.multiply(target_patches, rand_scale)
            image_aug = tmp_patches / np.max(tmp_patches, axis=(2, 3), keepdims=True)
            image_aug = seq.augment_images(image_aug)
            image_trans = np.transpose(image_aug, (0, 3, 1, 2))
        else:
            image_trans = np.transpose(target_patches, (0, 3, 1, 2)) / 256
        return image_trans

    def set_train(self):
        self.train = True

    def set_test(self):
        self.train = False

    def get_train(self):
        ret = []
        for trainid in self.train_id:
            ret.append((
                self.patches[trainid],
                (self.labels[trainid], self.eids[trainid])
            ))
        return ret

    def get_test(self):
        ret = []
        for testid in self.test_id:
            ret.append((
                self.patches[testid],
                (self.labels[testid], self.eids[testid])
            ))
        return ret


    # def get_train(self):
    #     ret = []
    #     for trainid in range(len(self.train_labels)):
    #         ret.append((
    #             self.train_patches[trainid],
    #             (self.train_labels[trainid], self.train_eids[trainid])
    #         ))
    #     return ret
    #
    # def get_test(self):
    #     ret = []
    #     for testid in range(len(self.test_labels)):
    #         ret.append((
    #             self.test_patches[testid],
    #             (self.test_labels[testid], self.test_eids[testid])
    #         ))
    #     return ret


if __name__ == "__main__":
    # _test_get_classification_folder()
    # _test_get_patch()
    ccb = BCBags()
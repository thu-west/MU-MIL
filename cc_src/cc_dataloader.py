import os

import numpy as np
import torch
import torch.utils.data as data_utils
import imgaug as ia
from imgaug import augmenters as iaa
import scipy.io as sio
from scipy.stats import ortho_group
import imageio

detection_dir = "../datasets/ColonCancer/Detection"
classification_dir = "../datasets/ColonCancer/Classification"

num_range = list(range(1, 101))

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
    iaa.ContrastNormalization((0.75, 1.5)),
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
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order



HCD = np.array([
    [1.88, -0.07, -0.60],
    [-1.02, 1.13, -0.48],
    [-0.55, -0.13, 1.57]
])

def get_classification_folder(num):
    class_dir = os.path.join(classification_dir, "img%d" % num)
    # load epithelial
    efile = os.path.join(class_dir, "img%d_epithelial.mat" % num)
    e_xy = sio.loadmat(efile)['detection']

    # load fibroblast
    ffile = os.path.join(class_dir, "img%d_fibroblast.mat" % num)
    f_xy = sio.loadmat(ffile)['detection']

    # load inflammatory
    ifile = os.path.join(class_dir, "img%d_inflammatory.mat" % num)
    i_xy = sio.loadmat(ifile)['detection']

    # load others
    ofile = os.path.join(class_dir, "img%d_others.mat" % num)
    o_xy = sio.loadmat(ofile)['detection']

    img = np.asarray(imageio.imread(os.path.join(class_dir, "img%d.bmp" % num)), dtype=np.uint8)
    return e_xy, f_xy, i_xy, o_xy, img


def get_detection_folder(num):
    ddir = os.path.join(detection_dir, "img%d" % num)
    dfile = os.path.join(ddir, "img%d_detection.mat" % num)
    d_xy = np.asarray(sio.loadmat(dfile)['detection'])
    img = imageio.imread(os.path.join(ddir, "img%d.bmp" % num))
    return d_xy, img


def get_patch(img, xy):
    x, y = xy
    x_id = int(x)
    y_id = int(y)
    if x_id - 14 < 0:
        xmin = 0
        xmax = xmin + 27
    elif x_id + 13 > 499:
        xmax = 499
        xmin = xmax - 27
    else:
        xmin = x_id - 14
        xmax = x_id + 13

    if y_id - 14 < 0:
        ymin = 0
        ymax = ymin + 27
    elif y_id + 13 > 499:
        ymax = 499
        ymin = ymax - 27
    else:
        ymin = y_id - 14
        ymax = y_id + 13

    patch = img[xmin:xmax, ymin:ymax, :]
    assert patch.shape == (27, 27, 3)

    patch = patch.reshape(27, 27, 1, 3)
    hcpatch = np.sum(np.multiply(patch, HCD), -1)

    # hcpatch = np.transpose(hcpatch, (2, 0, 1))
    hcpatch = hcpatch.reshape(1, 27, 27, 3).astype(np.uint8)

    return hcpatch


class CCBags(data_utils.Dataset):
    def __init__(self, aug_times=0):
        self.labels = []
        self.patches = []
        self.eids = []

        self.o_labels = []
        self.o_patches = []
        self.o_eids = []

        for n in num_range:
            # dxy, img = get_detection_folder(n)
            exy, fxy, ixy, oxy, img = get_classification_folder(n)
            self.o_labels.append(int(len(exy) > 0))
            img_patch = []
            e_id = []
            if len(exy) > 0:
                for xy in exy:
                    p = get_patch(img, xy)
                    img_patch.append(p)
                    e_id.append(1)
            for xy in fxy:
                p = get_patch(img, xy)
                img_patch.append(p)
                e_id.append(0)
            for xy in ixy:
                p = get_patch(img, xy)
                img_patch.append(p)
                e_id.append(0)
            for xy in oxy:
                p = get_patch(img, xy)
                img_patch.append(p)
                e_id.append(0)
            if len(img_patch) > 1:
                patches = np.concatenate(img_patch, 0)
            else:
                patches = img_patch[0].reshape(1, 27, 27, 3)
            self.o_patches.append(patches)
            self.o_eids.append(e_id)

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

        devide = int(len(self.labels)*0.7)
        self.train_id = shuffled_ids[:devide]
        self.test_id = shuffled_ids[devide:]

    def data_argument(self, Trans=True):
        r_labels, r_eids, r_patches = [], [], []
        for trainid in range(100):
            target_patches = self.o_patches[trainid]
            # if np.random.uniform(0, 1, 1) > 0.5:
            #     return target_patches
            # else:
            #     rand_scale = np.random.normal(1, 0.1, target_patches.shape)
            #     tmp_patches = np.multiply(target_patches, rand_scale)
            #     image = tmp_patches / np.max(tmp_patches, axis=(2, 3), keepdims=True) * 256
            #     return image
            if Trans:
                rand_scale = np.random.normal(1, 0.1, target_patches.shape)
                tmp_patches = np.multiply(target_patches, rand_scale)
                image_aug = tmp_patches / np.max(tmp_patches, axis=(2, 3), keepdims=True)
                image_aug = seq.augment_images(image_aug)
                image_trans = np.transpose(image_aug, (0, 3, 1, 2))
            else:
                image_trans = np.transpose(target_patches, (0, 3, 1, 2))/256
            r_labels.append(self.o_labels[trainid])
            r_eids.append(self.o_eids[trainid])
            r_patches.append(image_trans)
        return r_labels, r_eids, r_patches

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


def _test_get_classification_folder():
    num_e, num_f, num_i, num_o = 0, 0, 0, 0
    for n in num_range:
        print(n)
        e, f, i, o, img = get_classification_folder(n)

        num_e += e.shape[0]
        num_f += f.shape[0]
        num_i += i.shape[0]
        num_o += o.shape[0]

        # print(e, f, i, o)
        # print(img) # 500, 500, 3
        # e, f, i, o # num, 2
    print(num_e, num_f, num_i, num_o)
    return


def _test_get_patch():
    e, f, i, o, img = get_classification_folder(1)
    p = get_patch(img, f[0,:])
    print(p)
    print(p.shape)


if __name__ == "__main__":
    # _test_get_classification_folder()
    # _test_get_patch()
    ccb = CCBags()
    print(ccb.labels)
    ls = []
    for patches in ccb.all_patches:
        ls.append(len(patches))
        print(patches.shape)
    print(np.mean(ls), np.std(ls))
from __future__ import print_function

import sys
sys.path.append("../src")
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

import argparse
from sklearn import metrics
from scipy.stats import skew
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import pandas as pd

from model import *

# Training settings
np.set_printoptions(4)

parser = argparse.ArgumentParser(description='Multi-Preference Model')
# training parameters
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=0.0005, metavar='R',
                    help='weight decay')
parser.add_argument('--dim', type=int, default=5, metavar="K", help="dimension of the attention")

# MNIST setting, only useful for mnist bags dataset
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=10, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=500, metavar='NTest',
                    help='number of bags in test set')

# system setting
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='choose which gpu')
parser.add_argument('--inst_thr', type=float, default=0.5,
                    help='instance threshold')
# dataset selection
parser.add_argument('--dataset', type=str, default="cc", help="dataset in [cc, bc, mb, musk1, musk2, fox, tiger, elephant]")
parser.add_argument('--attention', type=str, default="att", help="attention in [att, mu, datt, mhatt]")

column_names = ["epoch", "train loss", "train err", "Dloss", "Dscore", "train skew",
                "test loss", "test err",
                "bag AUC", "bag p", "bag r", "bag f", "bag a",
                "ins AUC", "ins p", "ins r", "ins f ", "ins a", "test skew"]

def get_data_loader(dataset_name):
    if dataset_name == "mb":
        from mnist_dataloader import MnistBags
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                       mean_bag_length=args.mean_bag_length,
                                                       var_bag_length=args.var_bag_length,
                                                       num_bag=args.num_bags_train,
                                                       seed=np.random.randint(0, 10),
                                                       train=True),
                                             batch_size=1,
                                             shuffle=True,
                                             **loader_kwargs)

        test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                      mean_bag_length=args.mean_bag_length,
                                                      var_bag_length=args.var_bag_length,
                                                      num_bag=args.num_bags_test,
                                                      seed=np.random.randint(0, 10),
                                                      train=False),
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
        return train_loader, test_loader

    if dataset_name == "cc":
        from cc_dataloader import CCBags
        ccb = CCBags(aug_times=2)
        train_loader = data_utils.DataLoader(ccb.get_train(), batch_size=1, shuffle=True)
        test_loader = data_utils.DataLoader(ccb.get_test(), batch_size=1, shuffle=False)
        return train_loader, test_loader

    if dataset_name == "bc":
        from bc_dataloader import BCBags

        # seed = int(np.random.rand() * 256)
        seed = 123
        bcb = BCBags(seed=seed)
        train_loader = data_utils.DataLoader(bcb.get_train(), batch_size=1, shuffle=True)
        test_loader = data_utils.DataLoader(bcb.get_test(), batch_size=1, shuffle=False)
        return train_loader, test_loader

    if dataset_name in ["musk1", "musk2", "fox", "tiger", "elephant"]:
        from classic_dataloader import ClassicBag
        c5b = ClassicBag(name=dataset_name)
        train_loader = data_utils.DataLoader(c5b.get_train(), batch_size=1, shuffle=True)
        test_loader = data_utils.DataLoader(c5b.get_test(), batch_size=1, shuffle=False)
        return train_loader, test_loader


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0
    d_loss = 0
    d_score = 0
    all_score = []
    num_bag_labels = 0
    # train_data = train_loader.get_train()
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.float().cuda(), bag_label.float().cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, error, y_prob, Y_hat, score, feature = model.bag_eval(data, bag_label)
        train_loss += loss.cpu().data.numpy().reshape(-1)[0]
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        if bag_label > 0:
            inside_bag_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score) + 1e-5)
            all_score += inside_bag_score.detach().cpu().numpy().ravel().tolist()
            num_bag_labels += 1
            bag_d_loss, bag_d_score = discriminator_score(feature, instance_labels)
            d_loss += bag_d_loss
            d_score += bag_d_score
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    d_loss /= num_bag_labels
    d_score /= num_bag_labels
    train_skew = skew(all_score)

    info_column_value = [epoch, train_loss, train_error, d_loss, d_score, train_skew]
    return info_column_value


def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    show_count = 0
    all_score = []
    bag_label_list = []
    bag_score_list = []
    bag_preds_list = []
    instance_label_list = []
    instance_score_list = []
    instance_preds_list = []

    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if type(instance_labels) is torch.Tensor:
            instance_labels = instance_labels.cpu().view(-1).data.numpy().tolist()
        if args.cuda:
            data, bag_label = data.float().cuda(), bag_label.float().cuda()

        # above is the data loading procedure, should be customized by some input

        loss, error, y_prob, Y_hat, score, feature = model.bag_eval(data, bag_label)

        test_loss += loss.data[0]
        test_error += error

        bag_label_list.append(bag_label.cpu().detach().numpy())
        bag_score_list.append(y_prob.cpu().detach().numpy()[0])
        bag_preds_list.append(int(bag_score_list[-1]>0.5))

        if bag_preds_list[-1] > 0:
            inside_bag_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score) + 1e-5)
            all_score += inside_bag_score.detach().cpu().numpy().ravel().tolist()
            instance_label_list += instance_labels
            instance_preds_list += (inside_bag_score > args.inst_thr).cpu().data.numpy().reshape(-1).tolist()
            instance_score_list += inside_bag_score.cpu().data.numpy().reshape(-1).tolist()

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    test_skew = skew(all_score)

    P, R, F, _ = metrics.precision_recall_fscore_support(bag_label_list, bag_preds_list, average='binary')
    A = metrics.accuracy_score(bag_label_list, bag_preds_list)
    bag_auc = metrics.roc_auc_score(bag_label_list, bag_score_list)

    try:
        p, r, f, _ = metrics.precision_recall_fscore_support(instance_label_list, instance_preds_list, average='binary')
        a = metrics.accuracy_score(instance_label_list, instance_preds_list)
        ins_auc = metrics.roc_auc_score(instance_label_list, instance_score_list)
    except:
        p, r, f, a = 0, 0, 0, 0
        ins_auc = 0
    # print(np.linalg.norm(model.attention_model.deeper_nets.weight.detach().cpu().numpy()))
    critic = test_error.cpu().data+test_loss.cpu().data

    info_column_value = [test_loss.cpu().numpy()[0], test_error,
                         bag_auc, P, R, F, A,
                         ins_auc, p, r, f, a,
                         test_skew]

    return critic, info_column_value, instance_label_list, instance_score_list


def evaluate_cmd(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Init Model')

    if args.dataset == "mb":
        model = ModelFramework(encoder=Encoder_MB, attention_model=PIO_dict[args.attention], K=args.dim)
    elif args.dataset == "cc":
        model = ModelFramework(encoder=Encoder_CC, attention_model=PIO_dict[args.attention], K=args.dim)
    elif args.dataset in ["musk1", "musk2"]:
        data_encoder = Encoder_Classic1
        model = ModelFramework(encoder=Encoder_Classic1, attention_model=PIO_dict[args.attention], L=64, K=args.dim)
    elif args.dataset in ["fox", "tiger", "elephant"]:
        model = ModelFramework(encoder=Encoder_Classic2, attention_model=PIO_dict[args.attention], L=64, K=args.dim)
    elif args.dataset == "bc":
        model = ModelFramework(encoder=Encoder_BC, attention_model=PIO_dict[args.attention], K=args.dim)
    else:
        print("dataset %s not implemented" % args.dataset)
        raise NotImplementedError

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    print('Load Train and Test Set')

    train_loader, test_loader = get_data_loader(args.dataset)

    critic = 100
    stopping_value = None

    df = pd.DataFrame(columns=column_names)
    label, score = [], []

    def print_record(values, print_name=False, end=""):
        col_number = len(values)
        name_format = "{:5}" + "|{:10}" * (col_number-1)
        value_format = "{:5d}" + "|{:10.4f}" * (col_number-1)
        if print_name:
            print(name_format.format(*values), end=end+"\n")
        else:
            print(value_format.format(*values), end=end+"\n")

    print_record(column_names, True)

    for epoch in range(1, args.epochs + 1):
        train_col_value = train(epoch, model, optimizer, train_loader)
        c, test_col_value, ins_label, ins_score = test(model, test_loader)

        col_values = train_col_value + test_col_value
        df.loc[epoch-1] = col_values

        if c < critic:
            stopping_value = col_values
            critic = c
            print_record(col_values, end="*")
            label, score = ins_label, ins_score
        else:
            print_record(col_values)

    print()
    print_record(column_names, True)
    print_record(stopping_value)


    pos_score = [s for s, l in zip(score, label) if l == 1]
    neg_score = [s for s, l in zip(score, label) if l == 0]

    plt.hist([pos_score, neg_score], 20, histtype='bar')
    plt.legend(["positive", "negative"])
    plt.savefig("hist-%d.png" % args.dim)


    return df, stopping_value


if __name__ == "__main__":
    args = parser.parse_args()
    # evaluate_cmd()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    df, stopping_value = evaluate_cmd(args)
    model_str = "{}-{}-{}".format(args.dataset, args.attention, args.dim)
    df.to_csv(model_str + ".csv")
    with open('best_records.csv', mode='at') as f:
        tofile = " & ".join([model_str] + stopping_value)
        f.writelines(tofile + '\n')



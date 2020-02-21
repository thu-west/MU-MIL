from __future__ import print_function

import sys
sys.path.append("../src")
import os
import numpy as np

import argparse
from sklearn import metrics
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable


from cc_dataloader import CCBags
from model import ModelFramework, MultiDimAttention, MultiLayerAttention, Attention_CC, Encoder_CC, discriminator_score

# Training settings
np.set_printoptions(4)

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=0.0005, metavar='R',
                    help='weight decay')
parser.add_argument('--dim', type=int, default=15, metavar="K", help="dimension of the attention")
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=4, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=10, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=500, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='choose which gpu')
parser.add_argument('--inst_thr', type=float, default=0.5,
                    help='instance threshold')


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    d_loss = 0
    d_score = 0
    all_score = 0
    num_bag_labels = 0
    train_data = train_loader.get_train()
    for batch_idx, (data, label) in enumerate(train_data):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = torch.from_numpy(data).float().cuda(), torch.tensor(bag_label).float().cuda()
        data, bag_label = Variable(data), Variable(bag_label)
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
            num_bag_labels += 1
            bag_d_loss, bag_d_score = discriminator_score(feature, instance_labels)
            d_loss += bag_d_loss
            d_score += bag_d_score
    # calculate loss and error for epoch
    train_loss /= len(train_data)
    train_error /= len(train_data)
    d_loss /= num_bag_labels
    d_score /= num_bag_labels

    print('Epoch: {:2d}, Train Loss: {:.4f}, Train error: {:.4f}, d_loss {:.4f}, d_score {:.4f}'.format(
        epoch, train_loss, train_error, d_loss, d_score), end="|\t")


def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    show_count = 0
    bag_labels = []
    pred_probs = []
    test_data = test_loader.get_test()
    all_pos_bag_auc = 0
    instance_label_list = []
    instance_score_list = []
    instance_preds_list = []

    for batch_idx, (data, label) in enumerate(test_data):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = torch.from_numpy(data).float().cuda(), torch.tensor(bag_label).float().cuda()

        # above is the data loading procedure, should be customized by some input

        loss, error, y_prob, Y_hat, score, feature = model.bag_eval(data, bag_label)

        test_loss += loss.data[0]
        test_error += error

        bag_labels.append(bag_label.cpu().detach().numpy())
        pred_probs.append(y_prob.cpu().detach().numpy()[0])

        if bag_label > 0:
            inside_bag_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
            instance_label_list += instance_labels
            instance_preds_list += (inside_bag_score > args.inst_thr).cpu().data.numpy().reshape(-1).tolist()
            instance_score_list += inside_bag_score.cpu().data.numpy().reshape(-1).tolist()
            all_pos_bag_auc += metrics.roc_auc_score(instance_labels, inside_bag_score.cpu().data.numpy().reshape(-1))

    test_error /= len(test_data)
    test_loss /= len(test_data)

    # p, r, f, s = metrics.precision_recall_fscore_support(instance_label_list, instance_preds_list, average='binary')
    tp = sum([1 for l, p in zip(instance_label_list, instance_preds_list) if p == 1 and l == 1])
    ts = sum(instance_label_list)
    ps = sum(instance_preds_list)
    p = tp / ps
    r = tp / ts
    f = 2 * p * r / (p + r)

    label_score_pair = sorted(zip(instance_score_list, instance_label_list), key=lambda x: x[0], reverse=True)
    sorted_label = [l for s, l in label_score_pair]

    L = len(sorted_label)

    a = sum([1 for l, p in zip(instance_label_list, instance_preds_list) if p == l]) / L

    R = sum(sorted_label)
    p01 = sum(sorted_label[:int(0.01 * L)]) / int(0.01 * L)
    r01 = sum(sorted_label[:int(0.01 * L)]) / R
    p10 = sum(sorted_label[:int(0.10 * L)]) / int(0.10 * L)
    r10 = sum(sorted_label[:int(0.10 * L)]) / R
    p50 = sum(sorted_label[:int(0.50 * L)]) / int(0.50 * L)
    r50 = sum(sorted_label[:int(0.50 * L)]) / R

    bag_auc = metrics.roc_auc_score(bag_labels, pred_probs)
    ins_auc = metrics.roc_auc_score(instance_label_list, instance_preds_list)
    stat_str = 'Test Set, Loss: {:.4f}, Test error: {:.4f}, Test bag AUC {:.4f}' \
               '\n p01 {:.4f} r01 {:.4f} p10 {:.4f} r10 {:.4f} p50 {:.4f} r50 {:.4f}, p {:.4f}, r {:.4f}, f {:.4f} a {:.4f} auc {:.4f}'.format(
               test_loss.cpu().numpy()[0], test_error, bag_auc,
               p01, r01, p10, r10, p50, r50, p, r, f, a, ins_auc)
    print(stat_str)
    return bag_auc, test_error.cpu().data+test_loss.cpu().data


def evaluate_cmd(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_loader = CCBags(aug_times=1)

    print('Init Model')
    model = ModelFramework(encoder=Encoder_CC, attention_model=MultiDimAttention, K=args.dim)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    auc_at_best = 0
    highest_auc = 0
    c = 100

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, data_loader)
        auc, criteria = test(model, data_loader)
        if auc > highest_auc:
            highest_auc = auc
        if criteria < c:
            auc_at_best = auc
            c = criteria


    print("auc at best", auc_at_best)
    print("highest auc", highest_auc)

    return auc_at_best


if __name__ == "__main__":
    args = parser.parse_args()
    # evaluate_cmd()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    evaluate_cmd(args)

from __future__ import print_function

import sys
sys.path.append("../src")

import numpy as np

import argparse
from sklearn import metrics
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable


from classic_dataloader import ClassicBag
from model import ModelFramework, MultiDimAttention, Encoder_Classic1, Encoder_Classic2, discriminator_score

# Training settings
np.set_printoptions(4)

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=0.0005, metavar='R',
                    help='weight decay')
parser.add_argument('--dim', type=int, default=5, metavar="K", help="dimension of the attention")
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


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
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
    # calculate loss and error for epoch
    train_loss /= len(train_data)
    train_error /= len(train_data)

    print('Epoch: {:2d}, Train Loss: {:.4f}, Train error: {:.4f}'.format(
        epoch, train_loss, train_error), end="|\t")


def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    show_count = 0
    bag_labels = []
    pred_probs = []
    test_data = test_loader.get_test()
    all_pos_bag_auc = 0

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

    test_error /= len(test_data)
    test_loss /= len(test_data)

    bag_auc = metrics.roc_auc_score(bag_labels, pred_probs)
    print('\tTest Set, Loss: {:.4f}, Test error: {:.4f}, Test bag AUC {:.4f}'.format(
        test_loss.cpu().numpy()[0], test_error, bag_auc))
    return test_error, test_error+test_loss


def evaluate_cmd(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_loader = ClassicBag(name='elephant')

    print('Init Model')
    model = ModelFramework(encoder=Encoder_Classic, attention_model=MultiDimAttention, K=5, inputdim=230)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    err_at_best = 0
    lowest_err = 1
    c = 100

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, data_loader)
        err, criteria = test(model, data_loader)
        if err < lowest_err:
            lowest_err = err
        if criteria < c:
            err_at_best = err
            c = criteria

    print("auc at best", err_at_best)
    print("highest auc", lowest_err)

    return err_at_best


if __name__ == "__main__":
    args = parser.parse_args()
    # evaluate_cmd()

    evaluate_cmd(args)

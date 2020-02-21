from __future__ import print_function

import numpy as np


import sys
sys.path.append("../src")


import argparse
from sklearn import metrics
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable


from mnist_dataloader import MnistBags
from model import ModelFramework, MultiDimAttention, MultiLayerAttention, Attention_MB, Encoder_MB, discriminator_score



# Training settings
np.set_printoptions(4)

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                    help='weight decay')
parser.add_argument('--dim', type=int, default=1, metavar="K", help="dimension of the attention")
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=100, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=500, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    d_loss = 0
    d_score = 0
    all_score = 0
    num_bag_labels = 0
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
            num_bag_labels += 1
            bag_d_loss, bag_d_score = discriminator_score(feature, instance_labels)
            d_loss += bag_d_loss
            d_score += bag_d_score
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
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
    all_pos_bag_auc = 0

    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0].data[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        loss, error, y_prob, Y_hat, score, feature = model.bag_eval(data, bag_label)

        test_loss += loss.data[0]
        test_error += error

        bag_labels.append(bag_label.cpu().detach().numpy())
        pred_probs.append(y_prob.cpu().detach().numpy()[0])

        if bag_labels[-1] > 0 and instance_labels.view(-1).shape[0] > 1:
            inside_bag_score = (score - torch.min(score) + 1e-5) / (torch.max(score) - torch.min(score) + 1e-5)
            all_pos_bag_auc += metrics.roc_auc_score(instance_labels.cpu().data.numpy().reshape(-1),
                                                     inside_bag_score.cpu().data.numpy().reshape(-1))


    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    bag_auc = metrics.roc_auc_score(bag_labels, pred_probs)
    print('\tTest Set, Loss: {:.4f}, Test error: {:.4f}, Test bag AUC {:.4f} ins AUC {:.4f} '.format(
        test_loss.cpu().numpy()[0], test_error, bag_auc, all_pos_bag_auc / sum(bag_labels)))
    return bag_auc, test_error+test_loss


def evaluate_cmd(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                   mean_bag_length=args.mean_bag_length,
                                                   var_bag_length=args.var_bag_length,
                                                   num_bag=args.num_bags_train,
                                                   seed=args.seed,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                  mean_bag_length=args.mean_bag_length,
                                                  var_bag_length=args.var_bag_length,
                                                  num_bag=args.num_bags_test,
                                                  seed=args.seed,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    print('Init Model')
    model = ModelFramework(encoder=Encoder_MB, attention_model=MultiDimAttention, K=args.dim)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    auc_at_best = 0
    highest_auc = 0
    crit = 100

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, train_loader)
        auc, criteria = test(model, test_loader)
        if criteria < crit:
            auc_at_best = auc
            crit = criteria
        if auc > highest_auc:
            highest_auc = auc


    print("auc at best", auc_at_best)
    print("highest auc", highest_auc)

    return auc_at_best


# def evaluate_routine(dims, num_bags_train, mean_bag_length):
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#         print('\nGPU is ON!')
#
#     print('Load Train and Test Set')
#     loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#
#     results = []
#
#     for subeval in range(5):
#         # data loading
#         train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
#                                                        mean_bag_length=mean_bag_length,
#                                                        var_bag_length=mean_bag_length/5,
#                                                        num_bag=num_bags_train,
#                                                        seed=args.seed,
#                                                        train=True),
#                                              batch_size=1,
#                                              shuffle=True,
#                                              **loader_kwargs)
#
#         test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
#                                                       mean_bag_length=mean_bag_length,
#                                                       var_bag_length=mean_bag_length/5,
#                                                       num_bag=args.num_bags_test,
#                                                       seed=args.seed,
#                                                       train=False),
#                                             batch_size=1,
#                                             shuffle=False,
#                                             **loader_kwargs)
#
#
#
#         # original model
#         print("init model original model")
#         model_name = "origin"
#         model = Attention_MB()
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
#
#         if args.cuda:
#             model.cuda()
#         auc_at_best = 0
#         lowest_criteria = 100
#         highest_auc = 0
#
#         for epoch in range(1, args.epochs + 1):
#             train(epoch, model, optimizer, train_loader)
#             auc, criteria = test(model, test_loader)
#             if auc > highest_auc:
#                 highest_auc = auc
#             if criteria < lowest_criteria:
#                 lowest_criteria = criteria
#                 auc_at_best = auc
#
#         results.append((mean_bag_length, num_bags_train, model_name, highest_auc, auc_at_best))
#
#
#         for dim in dims:
#             print('Init Multi-Layer Attention Model @ K = ', dim)
#             model_name = "MLA%d@%deval" % (dim, subeval)
#             model = ModelFramework(encoder=Encoder_MB, attention_model=MultiLayerAttention, K=dim)
#             optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
#
#             if args.cuda:
#                 model.cuda()
#             auc_at_best = 0
#             lowest_criteria = 100
#             highest_auc = 0
#
#             for epoch in range(1, args.epochs + 1):
#                 train(epoch, model, optimizer, train_loader)
#                 auc, criteria = test(model, test_loader)
#                 if auc > highest_auc:
#                     highest_auc = auc
#                 if criteria < lowest_criteria:
#                     lowest_criteria = criteria
#                     auc_at_best = auc
#
#             results.append((mean_bag_length, num_bags_train, model_name, highest_auc, auc_at_best))
#
#
#         for dim in dims:
#             print('Init Multi-Dim Attention Model @ K = ', dim)
#             model_name = "MDA%d@%deval" % (dim, subeval)
#             model = ModelFramework(encoder=Encoder_MB, attention_model=MultiLayerAttention, K=dim)
#             optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
#
#             if args.cuda:
#                 model.cuda()
#             auc_at_best = 0
#             lowest_criteria = 100
#             highest_auc = 0
#
#             for epoch in range(1, args.epochs + 1):
#                 train(epoch, model, optimizer, train_loader)
#                 auc, criteria = test(model, test_loader)
#                 if auc > highest_auc:
#                     highest_auc = auc
#                 if criteria < lowest_criteria:
#                     lowest_criteria = criteria
#                     auc_at_best = auc
#
#             results.append((mean_bag_length, num_bags_train, model_name, highest_auc, auc_at_best))
#
#     return results

if __name__ == "__main__":
    args = parser.parse_args()
    evaluate_cmd(args)

    # def print_table(results):
    #     title_str = "{:20} | {:20} | {:20} | {:20} | {:20} |"
    #     content_str = "{:20f} | {:20f} | {:20} | {:20f} | {:20f} |"
    #     print(title_str.format("mean_bag_legnth", "num_bags_train", "model_name", "best_auc", "auc_at_best"))
    #     for mean_bag_length, num_bags_train, model_name, best_auc, auc_at_best in results:
    #         print(content_str.format(mean_bag_length, num_bags_train, model_name, best_auc, auc_at_best))

    # results = []
    # # dims = [1, 3, 5, 11, 23]
    # dims = [3, 5, 7, 9]
    # print_table(results)
    # for mean_bag_length in [5]:
    #     for num_bags_train in [10, 20, 50, 100]:
    #         print(mean_bag_length, num_bags_train)
    #         results += evaluate_routine(dims, num_bags_train, mean_bag_length)
    #         print_table(results)

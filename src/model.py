# all in one file for modeling

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class Encoder_CC(nn.Module):
    def __init__(self, L=512):
        """
        LeNet5, implemented by ICML 2018, applied to MNIST dataset
        :param L: the dimension of the latent feature vector
        """
        super(Encoder_CC, self).__init__()
        self.L = L
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.part2 = nn.Sequential(
            nn.Linear(48 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.part1(x)
        H = H.view(-1, 48 * 5 * 5)
        H = self.part2(H)
        return H


class Encoder_BC(nn.Module):
    def __init__(self, L=512):
        """
        LeNet5, implemented by ICML 2018, applied to MNIST dataset
        :param L: the dimension of the latent feature vector
        """
        super(Encoder_BC, self).__init__()
        self.L = L
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.part2 = nn.Sequential(
            nn.Linear(48 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.part1(x)
        H = H.view(-1, 48 * 6 * 6)
        H = self.part2(H)
        return H


class Encoder_Classic1(nn.Module):
    def __init__(self, L=500):
        super(Encoder_Classic1, self).__init__()
        self.L = L
        self.feature_extractor = nn.Sequential(
            nn.Linear(166, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        return self.feature_extractor(x)


class Encoder_Classic2(nn.Module):
    def __init__(self, L=500):
        super(Encoder_Classic2, self).__init__()
        self.L = L
        self.feature_extractor = nn.Sequential(
            nn.Linear(230, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        return self.feature_extractor(x)


class Encoder_MB(nn.Module):
    def __init__(self, L=500):
        """
        LeNet5, implemented by ICML 2018, applied to MNIST dataset
        :param L: the dimension of the latent feature vector
        """
        super(Encoder_MB, self).__init__()
        self.L = L
        self.part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.part2 = nn.Sequential(
            nn.Linear(50*4*4, self.L), nn.ReLU()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.part1(x)
        H = H.view(-1, 50*4*4)
        H = self.part2(H)
        return H


def merge_attention(attention_vector):
    """
    merge multiple scores into one
    :param scores: N x K, score vector of dim K
    :return: 1 x N, all scores are merged
    """
    score_vector = F.softmax(attention_vector, 0)
    target = torch.sum(torch.mul(score_vector, attention_vector), dim=0, keepdim=True)
    score = F.softmax(torch.mm(target, F.normalize(attention_vector, 1).t()), 1).view(1, -1)
    # score = F.softmax(torch.mm(target, attention_vector.t()), 1).view(1, -1)
    return score


def merge_attention_mean(attention_vector):
    score_vector = F.softmax(attention_vector, 0)
    score = torch.mean(score_vector, 1).view(1, -1)
    return score

def merge_attention_dynamic_routing(attention_vector):
    pass

def column_random_pick(attention_vector, prob=0.5):
    """
    :param attention_vector:
    :return:
    """
    K = attention_vector.shape[1]
    # if K < 2:
    #     return attention_vector
    # else:
    #     pick = np.random.uniform(low=0, high=1, size=K)
    #     column_id = []
    #     for i in range(K):
    #         if pick[i] < prob:
    #             column_id.append(i)
    #
    #     if len(column_id) < 1:
    #         column_id.append(0)
    #
    #     pick_tensor = torch.LongTensor(column_id).cuda()
    #     return attention_vector[:, pick_tensor]

    rand_tensor = torch.ones(K) * prob
    dropout_matrix = torch.diag(torch.bernoulli(rand_tensor)).cuda()
    return torch.mm(attention_vector, dropout_matrix)


def discriminator_score(features, labels, thr=0.5):
    """
    get the binary discriminator error
    :param features: N x K tensor
    :param labels: N x 1 tensor
    :return:
    """
    N, K = features.shape
    x = features.detach()
    if type(labels) is torch.Tensor:
        y = labels.view(-1).float().cuda()
    elif type(labels) is list:
        y = torch.tensor(labels).float().cuda()
    logistic = nn.Sequential(nn.Linear(K, 1), nn.Sigmoid()).cuda()
    loss_func = nn.BCELoss()

    def loss_and_pred(x):
        y_score = logistic(x).view(-1)
        loss = loss_func(y_score, y)
        return loss.cpu(), y_score.cpu().data.numpy().reshape(-1)

    opt = torch.optim.Adam(logistic.parameters())
    for i in range(100):
        opt.zero_grad()

        loss, y_score = loss_and_pred(x)

        loss.backward()
        opt.step()

    loss, y_score = loss_and_pred(x)

    loss_val = loss.cpu().data.numpy()

    score_val = metrics.accuracy_score(y.cpu().numpy().reshape(-1), y_score>0.5)
    return loss_val, score_val


class GATT(nn.Module):
    def __init__(self, D=128, K=1, L=512):
        super(GATT, self).__init__()
        print("use naive attention")
        self.D = D
        self.K = K
        self.L = L
        self.attention_emb = nn.Sequential(               # input N x L
            nn.Linear(self.L, self.D), nn.Tanh()          # Embedding N x D
        )
        self.attention_trans = nn.Linear(self.D, 1, bias=False)       # scores N x K
        self.gate1 = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

    def forward(self, x):
        E = self.attention_emb(x)
        GE = self.gate1(x).ge(0.5).float()
        E = torch.mul(E, GE)
        V = self.attention_trans(E)
        A = F.softmax(V, 0).view(1, -1)
        M = torch.mm(A, x)
        return M, A, E


class ATT(nn.Module):
    def __init__(self, D=128, K=1, L=512):
        super(ATT, self).__init__()
        print("use naive attention")
        self.D = D
        self.K = K
        self.L = L
        self.attention_emb = nn.Sequential(               # input N x L
            nn.Linear(self.L, self.D), nn.Tanh()          # Embedding N x D
        )
        self.attention_trans = nn.Linear(self.D, 1, bias=False)       # scores N x K

    def forward(self, x):
        E = self.attention_emb(x)
        V = self.attention_trans(E)
        A = F.softmax(V, 0).view(1, -1)
        M = torch.mm(A, x)
        return M, A, E


class MHATT(nn.Module):
    def __init__(self, D=128, K=1, L=512):
        super(MHATT, self).__init__()
        print("use multi-head attention")
        self.D = D
        self.K = K
        self.L = L
        self.attention_emb = nn.Sequential(               # input N x L
            nn.Linear(self.L, self.D), nn.Tanh()          # Embedding N x D
        )
        self.attention_trans = nn.Linear(self.D, self.K, bias=False)       # scores N x K

    def forward(self, x):
        E = self.attention_emb(x)
        V = self.attention_trans(E)
        A = F.softmax(V, 0).mean(dim=1).view(1, -1)
        M = torch.mm(A, x)
        return M, A, E


class DATT(nn.Module):
    def __init__(self, D=128, K=1, L=512):
        super(DATT, self).__init__()
        print("use deeper attention")
        self.D = D
        self.K = K
        self.L = L
        self.attention_emb = nn.Sequential(               # input N x L
            nn.Linear(self.L, self.D), nn.Tanh()          # Embedding N x D
        )
        self.attention_trans = nn.Linear(self.D, self.K, bias=False)       # scores N x K
        self.deeper_nets = nn.Linear(self.K, 1, bias=False)

    def forward(self, x):
        E = self.attention_emb(x)
        V = self.attention_trans(E)
        A = F.softmax(V, 0)
        V = self.deeper_nets(A)
        A = F.softmax(V, 0).view(1, -1)
        M = torch.mm(A, x)
        return M, A, E


class MultiDimAttention(nn.Module):
    def __init__(self, D=128, K=1, L=512):
        """
        attention model for multi instance learning
        :param D: the dimension of embedding that one instance will be transformed into
        :param K: the number of classification planes
        :param L: the dimension of the latent feature vector
        """
        super(MultiDimAttention, self).__init__()
        self.D = D
        self.K = K
        self.L = L
        self.attention_emb = nn.Sequential(               # input N x L
            nn.Linear(self.L, self.D), nn.Tanh(),         # Embedding N x D
            # nn.Linear(self.L, self.D), nn.Tanh()
        )
        self.attention_trans = nn.Linear(self.D, self.K)  # scores N x K
        # self.attention_trans = nn.Sequential(
        #     nn.Linear(self.D, self.K * 2), nn.Tanh(), nn.Linear(self.K * 2, self.K)
        # )

    def forward(self, x):
        """
        forward feed network
        :param x: the bags of size N where each instance feature is a vector of length L
        :return: the bag representation, a vector of length L
        """
        E = self.attention_emb(x)           # Transform the input feature vector to embedding vector of size D
        V = self.attention_trans(E)         # attention vector N x K for each instance
        if self.training:
            V = column_random_pick(V)
        A = merge_attention(V)
        M = torch.mm(A, x) # 1xL
        return M, A, E

PIO_dict = {
    "mu": MultiDimAttention,
    "att": ATT,
    "gatt": GATT,
    "mhatt": MHATT,
    "datt": DATT
}


class Classifier(nn.Module):
    def __init__(self, L=500, thr=0.5):
        super(Classifier, self).__init__()
        self.L = L
        self.thr = thr
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 1),
            # nn.Dropout(),
            nn.Sigmoid()
        )

    def forward(self, x):
        Y_prob = self.classifier(x)
        Y_hat = torch.ge(Y_prob, self.thr).float()
        return Y_prob, Y_hat


class ModelFramework(nn.Module):
    def __init__(self, encoder, attention_model, D=128, K=1, L=512, thr=0.5, inputdim=None):
        super(ModelFramework, self).__init__()
        self.D = D
        self.K = K
        self.L = L
        self.thr = thr
        try:
            self.feature_extractor = encoder(indim=166, L=L) if inputdim is None else encoder(indim=inputdim, L=L)
        except TypeError:
            self.feature_extractor = encoder(L=L)
        self.attention_model = attention_model(D=D, K=K, L=L) if K > 1 else NaiveAtt(D, K, L)
        self.classifier = Classifier(L=L, thr=thr)

    def forward(self, x):
        enc = self.feature_extractor(x)
        bag_feature, scores, feature = self.attention_model(enc)
        Y_prob, Y_hat = self.classifier(bag_feature)
        return Y_prob, Y_hat, scores, feature

    # AUXILIARY METHODS
    def bag_eval(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, score, feature = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data
        y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (Y * torch.log(y_prob) + (1. - Y) * torch.log(1. - y_prob))  # negative log bernoulli

        return loss, error, y_prob, Y_hat, score, feature

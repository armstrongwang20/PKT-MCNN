import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
import scipy.io as sciio
from FaultDataset2d import FaultDataset2d
import math

# Hyper-parameters for training
batch_size = 256
learning_rate = 1e-5
num_epoch = 250
gpu_ids = [3, 4, 5]
isMultiGpu = True
device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else "cpu")


def gen_coarselabel(fine_labels, relation):
    c_labels = []
    for i in range(len(fine_labels)):
        fl = int(fine_labels[i])
        cl = relation[fl]
        c_labels.append(cl)

    return np.array(c_labels)


def normalize_2darray(range_min, range_max, matrix):
    x = matrix.shape[0]
    y = matrix.shape[1]
    norm_mat = np.zeros(([x, y]))
    cur_max = max(map(max, matrix))
    cur_min = min(map(min, matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            norm_mat[i][j] = round(
                ((range_max - range_min) * (matrix[i][j] - cur_min) / (cur_max - cur_min)) + range_min)

    return norm_mat


def gen_superclass(data, fine_labels, num_class, num_clusters, isInit, model=None, dim_fc=128):
    from itertools import chain
    if isInit:
        print('Generate groups via original features')
        class_vec = cal_clsvec_init(data, fine_labels, num_class)
    else:
        print('Update groups via learned features')
        class_vec = cal_clsvec_iter(data, fine_labels, num_class, model, dim_fc)
    aff_mat = np.zeros([num_class, num_class])
    for a in range(0, num_class - 1):
        for b in range(a + 1, num_class):
            distance = np.linalg.norm(class_vec[a] - class_vec[b])
            aff_mat[a, b] = distance
            aff_mat[b, a] = aff_mat[a, b]
    # aff_mat = normalize_2darray(0, 1, aff_mat)
    beta = 0.1
    aff_mat = np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_class):
        aff_mat[i, i] = 0.0001
    sc = SpectralClustering(num_clusters, affinity='precomputed', assign_labels='discretize')  #
    groups = sc.fit_predict(aff_mat)

    return groups


def cal_clsvec_init(data, fine_labels, num_class):
    class_vec = np.zeros([num_class, data.shape[1]])
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels) if x == i]
        sigma_cls = np.zeros([data.shape[0], data.shape[1]])
        for m in range(len(idx)):
            s = data[:, :, idx[m]]
            avg_s = sum(s) / len(s)
            sigma_cls += avg_s
        vec = sum(sigma_cls) / len(idx)
        class_vec[i] = vec

    return class_vec


def cal_clsvec_iter(data, fine_labels, num_class, model, dim_fc):
    class_vec = np.zeros([num_class, dim_fc])
    # Generate features
    feas = np.zeros([data.shape[-1], dim_fc])
    x_in = torch.tensor(data, dtype=torch.float32).to(device)
    for i in range(x_in.shape[-1]):
        x = torch.unsqueeze(x_in[:, :, i], 0)
        x = torch.unsqueeze(x, 0)
        fea, out1, out2 = model(x)
        feas[i, :] = fea.cpu().detach().numpy()
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels) if x == i]
        s_vec = [m for m in feas[idx]]
        vec = sum(s_vec) / len(s_vec)
        class_vec[i] = vec

        return class_vec


def get_lambda(cur_epoch, coarse_train_ep, fine_train_ep):
    if cur_epoch > num_epoch - fine_train_ep - 1:
        my_lambda = 0
    elif cur_epoch < coarse_train_ep:
        my_lambda = 1
    else:
        my_lambda = 1 - ((cur_epoch + 1 - coarse_train_ep) / (num_epoch - fine_train_ep)) ** 2

    return my_lambda


def get_scalingfac(num1, num2):
    s1 = int(math.floor(math.log10(num1)))
    s2 = int(math.floor(math.log10(num2)))
    scale = 10 ** (s1 - s2)
    return scale


def toGPU(model):
    if isMultiGpu:
        model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
    else:
        model = model.to(device)

    return model


def train(epoch, coarse_ep, fine_ep):
    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_acc_1 = 0.0
    running_acc_2 = 0.0
    model.train()
    my_lambda = get_lambda(epoch, coarse_train_ep=coarse_ep, fine_train_ep=fine_ep)
    for data, target in train_loader:
        sample_train, label_train_f = data.to(device), torch.squeeze(target).long().to(device)
        label_train_c = torch.Tensor(gen_coarselabel(label_train_f, relation)).long().to(device)
        fea, out1, out2 = model(sample_train)
        if my_lambda == 0:
            loss1 = criterion(out1, label_train_c).detach().to(device)
            loss = criterion(out2, label_train_f).to(device)
            loss2 = loss
        elif my_lambda == 1:
            loss2 = criterion(out2, label_train_f).detach().to(device)
            loss = criterion(out1, label_train_c).to(device)
            loss1 = loss
        else:
            loss1 = criterion(out1, label_train_c).to(device)
            loss2 = criterion(out2, label_train_f).to(device)
            scale = get_scalingfac(loss1, loss2)
            loss = my_lambda * loss1 + (1 - my_lambda) * scale * loss2
        _, pred_c = out1.max(dim=1)
        _, pred_f = out2.max(dim=1)
        running_loss += loss.item()
        running_loss_1 += loss1.item()
        running_loss_2 += loss2.item()
        acc_batch_1 = (pred_c == label_train_c).sum()
        running_acc_1 += acc_batch_1.item()
        acc_batch_2 = (pred_f == label_train_f).sum()
        running_acc_2 += acc_batch_2.item()
        # Back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ========================= Log ======================
    print('Training...Finish {} epoch, Loss: {:.6f}, Loss1: {:.6f}, Acc1: {:.6f},'
          ' Loss2: {:.6f}, Acc2: {:.6f}'.format(epoch + 1, running_loss / num_train, running_loss_1 / num_train,
                                                running_acc_1 / num_train, running_loss_2 / num_train,
                                                running_acc_2 / num_train))
    f2.write('Training...Finish {} epoch, Loss: {:.6f}, Loss1: {:.6f}, Acc1: {:.6f},'
             ' Loss2: {:.6f}, Acc2: {:.6f}'.format(epoch + 1, running_loss / num_train,
                                                   running_loss_1 / num_train,
                                                   running_acc_1 / num_train, running_loss_2 / num_train,
                                                   running_acc_2 / num_train))


def test(epoch, coarse_ep, fine_ep):
    my_lambda = get_lambda(epoch, coarse_train_ep=coarse_ep, fine_train_ep=fine_ep)
    with torch.no_grad():
        eval_loss = 0.0
        eval_loss_1 = 0.0
        eval_loss_2 = 0.0
        eval_acc_1 = 0.0
        eval_acc_2 = 0.0
        model.eval()
        for data, target in test_loader:
            sample_test, label_test_f = data.to(device), torch.squeeze(target).long().to(device)
            label_test_c = torch.Tensor(gen_coarselabel(label_test_f, relation)).long().to(device)
            fea, out1, out2 = model(sample_test)
            _, pred_c = out1.max(dim=1)
            loss1 = criterion(out1, label_test_c)
            _, pred_f = out2.max(dim=1)
            loss2 = criterion(out2, label_test_f)
            scale = get_scalingfac(loss1, loss2)
            loss = my_lambda * loss1 + (1 - my_lambda) * scale * loss2
            eval_loss += loss.item()
            eval_loss_1 += loss1.item()
            eval_loss_2 += loss2.item()
            acc_batch_1 = (pred_c == label_test_c).sum()
            eval_acc_1 += acc_batch_1.item()
            acc_batch_2 = (pred_f == label_test_f).sum()
            eval_acc_2 += acc_batch_2.item()
        print('Test...Finish {} epoch, Loss: {:.6f}, Loss1: {:.6f}, Acc1: {:.6f},'
              ' Loss2: {:.6f}, Acc2: {:.6f}'.format(epoch + 1, eval_loss / num_test, eval_loss_1 / num_test,
                                                    eval_acc_1 / num_test, eval_loss_2 / num_test,
                                                    eval_acc_2 / num_test))
        f.write('Test...Finish {} epoch, Loss: {:.6f}, Loss1: {:.6f}, Acc1: {:.6f},'
                ' Loss2: {:.6f}, Acc2: {:.6f}'.format(epoch + 1, eval_loss / num_test, eval_loss_1 / num_test,
                                                      eval_acc_1 / num_test, eval_loss_2 / num_test,
                                                      eval_acc_2 / num_test))
        acc = eval_acc_2 / num_test

    return acc


def mkdir(path):

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


data = ''
data = sciio.loadmat(data)
train_set = data.get('x_train')
train_label = data.get('y_train')
test_set = data.get('x_valid')
test_label = data.get('y_valid')
train_label = train_label - 1
test_label = test_label - 1
data_train = FaultDataset2d(train_set, train_label)
data_test = FaultDataset2d(test_set, test_label)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=100, shuffle=False)

num_train = len(data_train)
num_test = len(data_test)
coarse_ep = 100
fine_ep = 150
best_ep = 0
num_class = 66
if __name__ == '__main__':
    model_num = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    for m in range(len(model_num)):
        model_file = __import__('Model_FaulDiag')
        model_cls_name = 'Net' + str(model_num[m]) + 'HT'
        Net = getattr(model_file, model_cls_name)
        candidate_num = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        for c in range(len(candidate_num)):
            best_acc = 0
            model = Net(candidate_num[c], num_class)
            if isMultiGpu:
                model = toGPU(model)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss().to(device)
            relation = gen_superclass(train_set, train_label, num_class=num_class, num_clusters=candidate_num[c], model=None,
                                      isInit=True)
            for epoch in range(num_epoch):
                print('epoch {}'.format(epoch + 1))
                print('*' * 10)
                train(epoch, coarse_ep, fine_ep, f2)
                acc = test(epoch, coarse_ep, fine_ep, f)
            print()
            print('BestAcc: {:.6f}, BestEP: {:.6f}'.format(best_acc, best_ep))
print()
print('End')

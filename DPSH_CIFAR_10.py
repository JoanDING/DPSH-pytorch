from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
import pdb
import pickle
from datetime import datetime
import argparse
import pdb

import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR

import CNN_model

def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit)
    if use_gpu:
        cnn_model = cnn_model.cuda()
    return cnn_model

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt

def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1

def DPSH_algo(param, gpu_ind=0):
    # parameters setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)

    DATA_DIR = 'data/{}'.format(param.dataset.upper())
    DATABASE_FILE = 'database_img.txt'
    TRAIN_FILE = 'train_img.txt'
    TEST_FILE = 'test_img.txt'

    DATABASE_LABEL = 'database_label.txt'
    TRAIN_LABEL = 'train_label.txt'
    TEST_LABEL = 'test_label.txt'

    epochs = param.max_epoch
    learning_rate = param.lr
    weight_decay = param.weight_decay
    model_name = param.model
    nclasses = param.nclasses
    use_gpu = torch.cuda.is_available()
    filename = param.filename
    lamda = param.lamda
    bit = param.bit
    #param['bit'] = bit
    #param['learning rate'] = learning_rate
    #param['model'] = model_name

    ### data processing
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if param.dataset == 'cifar-10':
        dset_database = DP.DatasetProcessingCIFAR_10(
            DATA_DIR, DATABASE_FILE, DATABASE_LABEL, transformations)

        dset_train = DP.DatasetProcessingCIFAR_10(
            DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)

        dset_test = DP.DatasetProcessingCIFAR_10(
            DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                              batch_size = param.batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size = param.batch_size,
                              shuffle=True,
                              num_workers=4
                             )

    test_loader = DataLoader(dset_test,
                             batch_size = param.batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit, use_gpu)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ### training phase
    # parameters setting
    B = torch.zeros(num_train, bit)
    U = torch.zeros(num_train, bit)
    train_labels = LoadLabel(TRAIN_LABEL, DATA_DIR)
    train_labels_onehot = EncodingOnehot(train_labels, nclasses)
    test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
    test_labels_onehot = EncodingOnehot(test_labels, nclasses)
    database_labels = LoadLabel(DATABASE_LABEL, DATA_DIR)
    database_labels_onehot = EncodingOnehot(database_labels, nclasses)

    train_loss = []
    map_record = []

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []

    Sim = CalcSim(train_labels_onehot, train_labels_onehot)

    for epoch in range(epochs):
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            train_label_onehot = EncodingOnehot(train_label, nclasses)
            if use_gpu:
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
            else:
                train_input, train_label = Variable(train_input), Variable(train_label)
            S = CalcSim(train_label_onehot, train_labels_onehot)

            model.zero_grad()
            train_outputs = model(train_input)
            for i, ind in enumerate(batch_ind):
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])

            Bbatch = torch.sign(train_outputs)
            if use_gpu:
                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / 2
                logloss = (Variable(S.cuda())*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))
            else:
                theta_x = train_outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S)*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))

            loss =  - logloss + lamda * regterm
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data)

            # print('[Training Phase][Epoch: %3d/%3d][Iteration: %3d/%3d] Loss: %3.5f' % \
            #       (epoch + 1, epochs, iter + 1, np.ceil(num_train / batch_size),loss.data[0]))
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)))
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

        l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
        totloss_record.append(l)
        totl1_record.append(l1)
        totl2_record.append(l2)
        t1_record.append(t1)

        #print('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l1, l2, t1))

        ### testing during epoch
        #qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
        #tB = torch.sign(B).numpy()
        #map_ = CalcHR.CalcMap(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy())
        train_loss.append(epoch_loss / len(train_loader))
        #map_record.append(map_)

        if epoch % param.test_epoch == 0:
            qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
            dB = GenerateCode(model, database_loader, num_database, bit, use_gpu)
            map_ = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
            map_5k =  CalcHR.CalcTopMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(),5000)
            map_50k =  CalcHR.CalcTopMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(),50000)

            print('[Test Phase ][Epoch: %3d/%3d] MAP(5k): %3.5f' % (epoch+1, epochs, map_5k))
            print('[Test Phase ][Epoch: %3d/%3d] MAP(50k): %3.5f' % (epoch+1, epochs, map_50k))
            print('[Test Phase ][Epoch: %3d/%3d] MAP(overall): %3.5f' % (epoch+1, epochs, map_))
        #print(len(train_loader))
    ### evaluation phase
    ## create binary code
    model.eval()
    qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
    dB = GenerateCode(model, database_loader, num_database, bit, use_gpu)

    map = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)

    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['train loss'] = train_loss
    result['map record'] = map_record
    result['map'] = map
    result['param'] = param
    result['total loss'] = totloss_record
    result['l1 loss'] = totl1_record
    result['l2 loss'] = totl2_record
    result['norm theta'] = t1_record
    result['filename'] = filename

    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu no')
    parser.add_argument('--lr', default = 0.05, type = float, help = 'learning rate')
    parser.add_argument('--weight_decay', default = 10e-5, type = float, help = 'weight_decay')
    parser.add_argument('--max_epoch', default = 200, type = int, help = 'max epoch num')
    parser.add_argument('--test_epoch', default = 10, type = int, help = 'max epoch num')
    parser.add_argument('--model', default = 'alexnet', type = str, help = 'model name')
    parser.add_argument('--batch_size', default = 128, type = int, help = 'batch size')
    parser.add_argument('--bit', default = 12, type = int, help = 'code length, can be [12, 24, 236, 48, 64, 128...]')
    parser.add_argument('--dataset', default = 'cifar-10', type = str, help = 'dataset name, can be only cifar-10 now')
    parser.add_argument('--lamda', default = 50, type = int, help = 'hyperparam lambda')
    opt = parser.parse_args()
    if opt.dataset == 'cifar-10':
        opt.nclasses = 10

    filename = 'log/DPSH_' + str(opt.bit) + 'bits_{}_'.format(opt.dataset) + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
    opt.filename = filename
    result = DPSH_algo(opt, opt.gpu)
    fp = open(result['filename'], 'wb')
    pickle.dump(result, fp)
    fp.close()


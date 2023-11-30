import torch
import torch.utils.data as dataf
import scipy.io as scio
import time
import numpy as np
from M2FNet import M2Fnet
import torch.nn as nn
import auxil
import csv
from auxil import yaml_load
from torch.utils.tensorboard import SummaryWriter

# tb_writer = SummaryWriter()  # tensorboard

tct = yaml_load('./cfg.yaml')
testSizeNumber = tct['testSizeNumber']
patchsize = tct['patch_size']
batchsize = tct['batch_size']
EPOCH = tct['epochs']
LR = float(tct['init_lr'])
FM = 16
datasetName = tct['train_dataset']
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
print('Separating the training set from the test set...')
datasetNames = ["Houston"]
FileName = ""
HSI = scio.loadmat('./datasets/HOUSTON2013/Houston2013_HSI.mat')
HSI = HSI['HSI']
HSI = HSI.astype(np.float32)

DSM = scio.loadmat('./datasets/HOUSTON2013/Houston2013_DSM.mat')
DSM = DSM['DSM']
DSM = DSM.astype(np.float32)

TR_map = scio.loadmat('./datasets/HOUSTON2013/Houston2013_TR.mat')
TR_map = TR_map['TR_map']
TE_map = scio.loadmat('./datasets/HOUSTON2013/Houston2013_TE.mat')
TE_map = TE_map['TE_map']

(m, n, z) = HSI.shape

for i in range(z):
    ma = np.max(HSI[:, :, i])
    mi = np.min(HSI[:, :, i])
    HSI[:, :, i] = (HSI[:, :, i] - mi) / (ma - mi)

    ma = np.max(DSM)
    mi = np.min(DSM)
    DSM = (DSM - mi) / (ma - mi)

pad_width = patchsize//2
k = 0

temp = HSI[:, :, 0]
pad_width = np.floor(patchsize / 2)
pad_width = np.int32(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2, n2] = temp2.shape
HSI2 = np.empty((m2, n2, z), dtype='float32')

for i in range(z):
    temp = HSI[:, :, i]
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    HSI2[:, :, i] = temp2

temp = DSM
temp2 = np.pad(temp, pad_width, 'symmetric')
DSM2 = temp2

temp = TR_map
temp2 = np.pad(temp, pad_width, 'symmetric')
TR_map2 = temp2

temp = TE_map
temp2 = np.pad(temp, pad_width, 'symmetric')
TE_map2 = temp2
# TestPatch = np.zeros(((m-6) * (n-6), patchsize, patchsize, z), dtype='float32')

[ind1_TR, ind2_TR] = np.where(TR_map != 0)
TrainNum = len(ind1_TR)
TrainLabel = np.zeros((TrainNum, 1), dtype='uint8')
TrainPatch_HSI = np.zeros((TrainNum, patchsize * patchsize * z), dtype='float32')
TrainPatch_DSM = np.zeros((TrainNum, patchsize * patchsize), dtype='float32')
k_TR = 0

for i in range(pad_width, m2 - pad_width):
    for j in range(pad_width, n2 - pad_width):
        patchlabel_TR = TR_map2[i, j]
        if (patchlabel_TR != 0):
            patch_HSI = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :]
            TrainPatch_HSI[k_TR, :] = np.reshape(patch_HSI,
                                                 [1, patch_HSI.shape[0] * patch_HSI.shape[1] * patch_HSI.shape[2]],
                                                 order="F")
            patch_DSM = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)]
            TrainPatch_DSM[k_TR, :] = np.reshape(patch_DSM, [1, patch_DSM.shape[0] * patch_DSM.shape[1]], order="F")
            TrainLabel[k_TR] = patchlabel_TR
            k_TR = k_TR + 1

[ind1_TE, ind2_TE] = np.where(TE_map != 0)
TestNum = len(ind1_TE)
TestPatch_HSI = np.zeros((TestNum, patchsize * patchsize * z), dtype='float32')
TestPatch_DSM = np.zeros((TestNum, patchsize * patchsize), dtype='float32')
TestLabel = np.zeros((TestNum, 1), dtype='uint8')
k_TE = 0

for i in range(pad_width, m2 - pad_width):
    for j in range(pad_width, n2 - pad_width):
        patchlabel_TE = TE_map2[i, j]
        if (patchlabel_TE != 0):
            patch_HSI = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :]
            TestPatch_HSI[k_TE, :] = np.reshape(patch_HSI,
                                                [1, patch_HSI.shape[0] * patch_HSI.shape[1] * patch_HSI.shape[2]],
                                                order="F")
            patch_DSM = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)]
            TestPatch_DSM[k_TE, :] = np.reshape(patch_DSM, [1, patch_DSM.shape[0] * patch_DSM.shape[1]], order="F")
            TestLabel[k_TE] = patchlabel_TE
            k_TE = k_TE + 1

TrainLabel= torch.as_tensor(torch.from_numpy(TrainLabel), dtype=torch.float32)
TestLabel= torch.as_tensor(torch.from_numpy(TestLabel), dtype=torch.float32)

TrainPatch1 = torch.tensor(TrainPatch_HSI.reshape(TrainPatch_HSI.shape[0], z, patchsize , patchsize )).to(torch.float32)
TrainPatch1_lidar = torch.tensor(TrainPatch_DSM.reshape(TrainPatch_DSM.shape[0], 1, patchsize , patchsize )).to(torch.float32)
TrainLabel1 = TrainLabel-1
TrainLabel1 = TrainLabel1.long().reshape(-1)
TestPatch1 = torch.tensor(TestPatch_HSI.reshape(TestPatch_HSI.shape[0], z, patchsize , patchsize )).to(torch.float32)
TestPatch1_lidar = torch.tensor(TestPatch_DSM.reshape(TestPatch_DSM.shape[0], 1, patchsize , patchsize )).to(torch.float32)
TestLabel1 = TestLabel-1
TestLabel1 = TestLabel1.long().reshape(-1)

dataset = dataf.TensorDataset(TrainPatch1,TrainPatch1_lidar,TrainLabel1)
Classes = len(np.unique(TrainLabel))
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers= 4)
NC = z

print("HSI Train data shape = ", TrainPatch1.shape)
print("Train label shape = ", TrainLabel1.shape)
print("HSI Test data shape = ", TestPatch1.shape)
print("Test label shape = ", TestLabel1.shape)
print("Number of Classes = ", Classes)

KAPPA = []
OA = []
AA = []
ELEMENT_ACC = np.zeros((1, Classes))

set_seed(42)

for iterNum in range(tct['go'],tct['end']+1):
    I = str(iterNum)
    Net = M2Fnet(FM, NC, Classes).cuda()
    optimizer = torch.optim.Adam(Net.parameters(), lr=LR,weight_decay=5e-3)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tct['step_size'], gamma=0.9)
    BestAcc = 0

    torch.cuda.synchronize()
    # train and test the designed model
    for epoch in range(EPOCH):
        for step, (hsi_data, lidar_data, Label) in enumerate(train_loader):
            # move train data to GPU
            hsi_data = hsi_data.cuda()
            lidar_data = lidar_data.cuda()
            Label = Label.cuda()

            out1 = Net(hsi_data, lidar_data)
            loss = loss_func(out1, Label)

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                Net.eval()
                pred_y = np.empty((len(TestLabel1)), dtype='float32')
                number = len(TestLabel1) // testSizeNumber
                for i in range(number):
                    temp = TestPatch1[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                    temp = temp.cuda()
                    temp_lidar = TestPatch1_lidar[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                    temp_lidar= temp_lidar.cuda()

                    temp2 = Net(temp,temp_lidar)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
                    del temp, temp2, temp3

                if (i + 1) * testSizeNumber < len(TestLabel1):
                    temp = TestPatch1[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                    temp = temp.cuda()
                    temp_lidar = TestPatch1_lidar[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                    temp_lidar = temp_lidar.cuda()

                    temp2 = Net(temp, temp_lidar)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cpu()
                    del temp, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % (accuracy*100))

                # save the parameters in network
                if accuracy > BestAcc:
                    BestAcc = accuracy
                    torch.save(Net.state_dict(), './datasets/'+datasetName+'/net_params_'+I+'_WCT.pkl')
                Net.train()

        scheduler.step()
    torch.cuda.synchronize()
    Net.load_state_dict(torch.load('./datasets/'+datasetName+'/net_params_'+I+'_WCT.pkl'))
    Net.eval()
    confusion, oa, each_acc, aa, kappa = auxil.reports(TestPatch1,TestPatch1_lidar,TestLabel1, datasetName, Net, testSizeNumber)
    print("OA AA, Kappa ACCclass", oa, aa, kappa, each_acc)



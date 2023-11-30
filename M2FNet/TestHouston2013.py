import torch
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib import colors
from M2FNet import M2Fnet
import torch.utils.data as dataf
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score,  cohen_kappa_score
from operator import truediv

#utils
import torch
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm
import re
import yaml

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for (data1, data2, labels) in tqdm(test_loader):
        data1, data2 = data1.to(device), data2.to(device)
        outputs = net(data1, data2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return  oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100

#data
def create_houston():
    tct = yaml_load('./cfg.yaml')
    patchsize = tct['patch_size']
    batchsize = tct['batch_size']

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
    y = TR_map + TE_map
    (m, n, z) = HSI.shape

    for i in range(z):
        ma = np.max(HSI[:, :, i])
        mi = np.min(HSI[:, :, i])
        HSI[:, :, i] = (HSI[:, :, i] - mi) / (ma - mi)

        ma = np.max(DSM)
        mi = np.min(DSM)
        DSM = (DSM - mi) / (ma - mi)

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

    [ind1_TR, ind2_TR] = np.where(TR_map != 0)
    TrainNum = len(ind1_TR)
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

    TestLabel= torch.as_tensor(torch.from_numpy(TestLabel), dtype=torch.float32)
    TestPatch1 = torch.tensor(TestPatch_HSI.reshape(TestPatch_HSI.shape[0], z, patchsize , patchsize )).to(torch.float32)
    TestPatch1_lidar = torch.tensor(TestPatch_DSM.reshape(TestPatch_DSM.shape[0], 1, patchsize , patchsize )).to(torch.float32)
    TestLabel1 = TestLabel-1
    TestLabel1 = TestLabel1.long().reshape(-1)

    TotalNum = TestNum + TrainNum
    TotalPatch_HSI = np.zeros((TotalNum, patchsize * patchsize * z), dtype='float32')
    TotalPatch_DSM = np.zeros((TotalNum, patchsize * patchsize), dtype='float32')
    TotalLabel = np.zeros((TotalNum, 1), dtype='uint8')
    TOTALmap2 = TR_map2 + TE_map2
    k_TE = 0
    pos_total =  np.zeros((TotalNum, 2), dtype='int32')
    for i in range(pad_width, m2 - pad_width):
        for j in range(pad_width, n2 - pad_width):
            patchlabel_TOTAL = TOTALmap2[i, j]
            if (patchlabel_TOTAL != 0):
                patch_HSI = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :]
                TotalPatch_HSI[k_TE, :] = np.reshape(patch_HSI,
                                                    [1, patch_HSI.shape[0] * patch_HSI.shape[1] * patch_HSI.shape[2]],
                                                    order="F")
                patch_DSM = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)]
                TotalPatch_DSM[k_TE, :] = np.reshape(patch_DSM, [1, patch_DSM.shape[0] * patch_DSM.shape[1]], order="F")
                TotalLabel[k_TE] = patchlabel_TOTAL
                # index = (i-pad_width)*m2 + (j-pad_width) * n2
                pos_total[k_TE:k_TE + 1,:] = [i-pad_width,j-pad_width]
                k_TE = k_TE + 1

    TotalPatch1 = torch.tensor(TotalPatch_HSI.reshape(TotalPatch_HSI.shape[0], z, patchsize, patchsize)).to(torch.float32)
    TotalPatch1_lidar = torch.tensor(TotalPatch_DSM.reshape(TotalPatch_DSM.shape[0], 1, patchsize, patchsize)).to(
        torch.float32)

    dataset02 = dataf.TensorDataset(TotalPatch1, TotalPatch1_lidar)
    total_loader = dataf.DataLoader(dataset02, batch_size=batchsize, shuffle=False, num_workers=4)
    dataset = dataf.TensorDataset(TestPatch1,TestPatch1_lidar,TestLabel1)
    test_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers= 4)
    return total_loader,test_loader,y,pos_total

if __name__ == '__main__':
    tct = yaml_load('./cfg.yaml')
    print(f"The dataset being tested is {tct['train_dataset']}")
    print('Reading model parameters...')
    device = torch.device(tct['device'])
    if tct['train_dataset'] == 'trento':
        net = M2Fnet(16, 63, 6).to(device)
    elif tct['train_dataset'] == 'aug':
        net = M2Fnet(16, 180, 7).to(device)
    elif tct['train_dataset']=='HOUSTON2013':
        net = M2Fnet(16, 144, 15).to(device)

    weights = tct['val_weights']
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weights).items()})
    print('Separating the training set from the test set...')
    if tct['train_dataset']=='HOUSTON2013':
        total_iter,test_iter,y,pos_total = create_houston()

    print('Testing the model...')
    y_pred_test, y_test = test(device, net, test_iter)
    oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    print('Output test results:')
    print(f'Kappa:{kappa},OA:{oa},AA:{aa}')

    # if tct['train_dataset']=='HOUSTON2013':
    #     generate_png_houston(total_iter, net, y, device, pos_total ,f"./{tct['train_dataset']}")








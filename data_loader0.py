import torch.utils.data as data
import h5py
import os
import numpy as np
import random

def get_sigloader(data_root, class_num, train_num_perclass, test_num_perclass, sp_len, sample_len, train_transform, test_transform):
    data = h5py.File(data_root, 'r')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(class_num):
        x = data[str(i)]
        n = x.shape[0]
        idx = np.random.permutation(np.arange(n))
        for j in range(train_num_perclass):
            sig = np.array([x[idx[j],:sp_len], x[idx[j],sp_len:]], dtype='float32')
            train_data.append(sig)
            train_label.append(i)
        for j in range(test_num_perclass):
            sig = np.array([x[idx[j+train_num_perclass],:sp_len], x[idx[j+train_num_perclass],sp_len:]], dtype='float32')
            test_data.append(sig)
            test_label.append(i)
    trainSigLoader = SigLoader(train_data, train_label, sp_len, sample_len, train_transform)
    testSigLoader = SigLoader(test_data, test_label, sp_len, sample_len, test_transform)
    return trainSigLoader, testSigLoader

def get_sigloader1(data_root, class_text,  test_num_perclass, sp_len, sample_len,  test_transform):
    data = h5py.File(data_root, 'r')
    # train_data = []
    # train_label = []
    test_data = []
    test_label = []
    for i in range(class_text):
        x = data[str(i)]
        n = x.shape[0]
        idx = np.random.permutation(np.arange(n))
        # for j in range(train_num_perclass):
        #     sig = np.array([x[idx[j], :sp_len], x[idx[j], sp_len:]], dtype='float32')
        #     train_data.append(sig)
        #     train_label.append(i)
        for j in range(test_num_perclass):
            sig = np.array([x[idx[j], :sp_len], x[idx[j], sp_len:]], dtype='float32')
            test_data.append(sig)
            test_label.append(i)
    # trainSigLoader = SigLoader(train_data, train_label, sp_len, sample_len, train_transform)
    testSigLoader = SigLoader(test_data, test_label, sp_len, sample_len, test_transform)
    return  testSigLoader

def get_sigloader2(data_root, inctement_class, train_num_perclass, test_num_perclass, sp_len, sample_len, train_transform, test_transform):
    data = h5py.File(data_root, 'r')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    class_num = 4
    for i in range(class_num):
        x = data[str(i)]
        n = x.shape[0]
        idx = np.random.permutation(np.arange(n))
        for j in range(150):
            sig = np.array([x[idx[j],:sp_len], x[idx[j],sp_len:]], dtype='float32')
            train_data.append(sig)
            train_label.append(i)
    for i in range(len(inctement_class)):
        x = data[str(inctement_class[i])]
        print('增量训练的类别是：',inctement_class[i])
        n = x.shape[0]
        idx = np.random.permutation(np.arange(n))
        for j in range(train_num_perclass):
            sig = np.array([x[idx[j],:sp_len], x[idx[j],sp_len:]], dtype='float32')
            train_data.append(sig)
            train_label.append(inctement_class[i])
        for j in range(test_num_perclass):
            sig = np.array([x[idx[j + train_num_perclass], :sp_len], x[idx[j + train_num_perclass], sp_len:]],
                           dtype='float32')
            test_data.append(sig)
            test_label.append(inctement_class[i])
    trainSigLoader = SigLoader(train_data, train_label, sp_len, sample_len, train_transform)
    testSigLoader = SigLoader(test_data, test_label, sp_len, sample_len, test_transform)
    return trainSigLoader, testSigLoader

def get_sigloader3(data_root, class_num, train_num_perclass, sp_len, sample_len, train_transform=None):
    data = h5py.File(data_root, 'r')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(class_num-1):
        x = data[str(i)]
        n = x.shape[0]
        idx = np.arange(n)
        # idx = np.random.permutation(np.arange(n))
        for j in range(2000):
            sig = np.array([x[idx[j],:sp_len], x[idx[j],sp_len:]], dtype='float32')
            train_data.append(sig)
            train_label.append(i)
    x = data[str(7)]
    n = x.shape[0]
    idx = np.arange(n)
    # idx = np.random.permutation(np.arange(n))
    for j in range(train_num_perclass):
        sig = np.array([x[idx[j], :sp_len], x[idx[j], sp_len:]], dtype='float32')
        train_data.append(sig)
        train_label.append(i)
        # for j in range(test_num_perclass):
        #     sig = np.array([x[idx[j+train_num_perclass],:sp_len], x[idx[j+train_num_perclass],sp_len:]], dtype='float32')
        #     test_data.append(sig)
        #     test_label.append(i)
    # trainSigLoader0 = SigLoader(train_data, train_label, sp_len, sample_len, None)
    trainSigLoader = SigLoader2(train_data, train_label, sp_len, sample_len, train_transform)
    # testSigLoader = SigLoader2(test_data, test_label, sp_len, sample_len, test_transform)
    return  trainSigLoader

class SigLoader(data.Dataset):
    def __init__(self, data, label, sp_len, sample_len, transform=None):
        self.transform = transform
        self.x = data
        self.y = label
        self.sp_len = sp_len
        self.sample_len = sample_len
        self.sp_num = sp_len // sample_len

    def __getitem__(self, item):
        imgs = self.x[item]
        labels = self.y[item]

        if self.transform is not None:
            imgsq = self.transform(imgs)
            # imgsk = self.transform(imgs)
        
        return imgsq, int(labels)

    def __len__(self):
        return len(self.x)


class SigLoader2(data.Dataset):
    def __init__(self, data, label, sp_len, sample_len, transform=None):
        self.transform = transform
        self.x = data
        self.y = label
        self.sp_len = sp_len
        self.sample_len = sample_len
        self.sp_num = sp_len // sample_len

    def __getitem__(self, item):
        imgs = self.x[item]
        labels = self.y[item]

        if self.transform is not None:
            imgsq = self.transform(imgs)
            imgsk = self.transform(imgs)

        return imgsq, imgsk, int(labels)

    def __len__(self):
        return len(self.x)


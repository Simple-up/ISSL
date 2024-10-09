import torch.utils.data as data
import h5py
# import os
import numpy as np
# import random


def get_sigloader(data_root, num, sp_len,sample_len, train_transform,test_transform):
    data_train = np.load(f"{data_root}/Dataset_50/X_train_{num}Class.npy")
    label_train = np.load(f"{data_root}/Dataset_50/Y_train_{num}Class.npy")
    data_test = np.load(f"{data_root}/Dataset_50/X_test_{num}Class.npy")
    label_test = np.load(f"{data_root}/Dataset_50/Y_test_{num}Class.npy")
    train_num = len(data_train)
    test_num = len(data_test)
    if train_num != len(label_train):
        raise ValueError("训练数据和标签长度不一致")
    if test_num != len(label_test):
        raise ValueError("测试数据和标签长度不一致")
    data_train = data_train.transpose(0, 2, 1)
    data_test = data_test.transpose(0, 2, 1)
    train_data = data_train.astype(np.float32)
    test_data = data_test.astype(np.float32)
    train_label = label_train.astype(np.uint8)
    test_label = label_test.astype(np.uint8)
    trainSigLoader = SigLoader(train_data, train_label, sp_len, sample_len, train_transform)
    testSigLoader = SigLoader(test_data, test_label, sp_len, sample_len, test_transform)
    return trainSigLoader, testSigLoader


def get_sigloader2(data_root, sp_len,sample_len, train_transform):
    num = 10
    data_train_1 = np.load(f"{data_root}/Dataset/X_train_{num}Class.npy")
    label_train_1 = np.load(f"{data_root}/Dataset/Y_train_{num}Class.npy")
    # data_test_1 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_1 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    train_num_1 = len(data_train_1)
    # test_num_1 = len(data_test_1)
    if train_num_1 != len(label_train_1):
        raise ValueError("十类训练数据和标签长度不一致")
    # if test_num_1 != len(label_test_1):
    #     raise ValueError("测试数据和标签长度不一致")
    num = 20
    data_train_2 = np.load(f"{data_root}/Dataset/X_train_{num}Class.npy")
    label_train_2 = np.load(f"{data_root}/Dataset/Y_train_{num}Class.npy")
    # data_test_2 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_2 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    train_num_2 = len(data_train_2)
    # test_num_2 = len(data_test_2)
    if train_num_2 != len(label_train_2):
        raise ValueError("二十类训练数据和标签长度不一致")
    # if test_num_2 != len(label_test_2):
    #     raise ValueError("测试数据和标签长度不一致")
    merged_train_data = np.concatenate((data_train_1, data_train_2), axis=0)
    merged_train_label = np.concatenate((label_train_1, label_train_2), axis=0)
    # merged_test_data = np.concatenate((data_test_1, data_test_2), axis=0)
    # merged_test_label = np.concatenate((label_test_1, label_test_2), axis=0)
    data_train =merged_train_data.transpose(0, 2, 1)
    # data_test = merged_test_data.transpose(0, 2, 1)
    train_data = data_train.astype(np.float32)
    # test_data = data_test.astype(np.float32)
    train_label = merged_train_label.astype(np.uint8)
    # test_label = merged_test_label.astype(np.uint8)
    trainSigLoader = SigLoader2(train_data, train_label, sp_len, sample_len, train_transform)
    # testSigLoader = SigLoader2(test_data, test_label, sp_len, sample_len, test_transform)
    return trainSigLoader


def get_sigloader_train(data_root, sp_len,sample_len, train_transform):
    num = 10
    data_train_1 = np.load(f"{data_root}/Dataset/X_train_{num}Class.npy")
    label_train_1 = np.load(f"{data_root}/Dataset/Y_train_{num}Class.npy")
    # data_test_1 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_1 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    train_num_1 = len(data_train_1)
    # test_num_1 = len(data_test_1)
    if train_num_1 != len(label_train_1):
        raise ValueError("十类训练数据和标签长度不一致")
    # if test_num_1 != len(label_test_1):
    #     raise ValueError("测试数据和标签长度不一致")
    num = 20
    data_train_2 = np.load(f"{data_root}/Dataset/X_train_{num}Class.npy")
    label_train_2 = np.load(f"{data_root}/Dataset/Y_train_{num}Class.npy")
    # data_test_2 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_2 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    train_num_2 = len(data_train_2)
    # test_num_2 = len(data_test_2)
    if train_num_2 != len(label_train_2):
        raise ValueError("二十类训练数据和标签长度不一致")
    # if test_num_2 != len(label_test_2):
    #     raise ValueError("测试数据和标签长度不一致")
    merged_train_data = np.concatenate((data_train_1, data_train_2), axis=0)
    merged_train_label = np.concatenate((label_train_1, label_train_2), axis=0)
    # merged_test_data = np.concatenate((data_test_1, data_test_2), axis=0)
    # merged_test_label = np.concatenate((label_test_1, label_test_2), axis=0)
    data_train =merged_train_data.transpose(0, 2, 1)
    # data_test = merged_test_data.transpose(0, 2, 1)
    train_data = data_train.astype(np.float32)
    # test_data = data_test.astype(np.float32)
    train_label = merged_train_label.astype(np.uint8)
    # test_label = merged_test_label.astype(np.uint8)
    trainSigLoader = SigLoader(train_data, train_label, sp_len, sample_len, train_transform)
    # testSigLoader = SigLoader2(test_data, test_label, sp_len, sample_len, test_transform)
    return trainSigLoader


def get_sigloader_test(data_root, sp_len,sample_len, test_transform):
    num = 10
    data_test_1 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    label_test_1 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    # data_test_1 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_1 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    test_num_1 = len(data_test_1)
    # test_num_1 = len(data_test_1)
    if test_num_1 != len(label_test_1):
        raise ValueError("十类测试数据和标签长度不一致")
    # if test_num_1 != len(label_test_1):
    #     raise ValueError("测试数据和标签长度不一致")
    num = 20
    data_test_2 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    label_test_2 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    # data_test_2 = np.load(f"{data_root}/Dataset/X_test_{num}Class.npy")
    # label_test_2 = np.load(f"{data_root}/Dataset/Y_test_{num}Class.npy")
    test_num_2 = len(data_test_2)
    # test_num_2 = len(data_test_2)
    if test_num_2 != len(label_test_2):
        raise ValueError("二十类训练数据和标签长度不一致")
    # if test_num_2 != len(label_test_2):
    #     raise ValueError("测试数据和标签长度不一致")
    # merged_train_data = np.concatenate((data_test_1, data_train_2), axis=0)
    # merged_train_label = np.concatenate((label_train_1, label_train_2), axis=0)
    merged_test_data = np.concatenate((data_test_1, data_test_2), axis=0)
    merged_test_label = np.concatenate((label_test_1, label_test_2), axis=0)
    # data_train =merged_train_data.transpose(0, 2, 1)
    data_test = merged_test_data.transpose(0, 2, 1)
    # train_data = data_train.astype(np.float32)
    test_data = data_test.astype(np.float32)
    # train_label = merged_train_label.astype(np.uint8)
    test_label = merged_test_label.astype(np.uint8)
    # trainSigLoader = SigLoader2(train_data, train_label, sp_len, sample_len, train_transform)
    testSigLoader = SigLoader(test_data, test_label, sp_len, sample_len, test_transform)
    return testSigLoader

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


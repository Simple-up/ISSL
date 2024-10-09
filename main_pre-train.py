# from model import Model
import torch
torch.backends.cudnn.benchmark=True
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import torchvision.datasets as dsets
# import torchvision.models as models
import torchvision.transforms as transforms
# from torch.autograd import Variable
import torch.optim as optim
# import torch.nn.functional as F
import resnet
import os
import argparse
# import data_loader0
# import time
import numpy as np
# import subprocess
# from numpy import random
import augmentations
# import copy

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# from data_loader import cifar10, cifar100
from data_loader import get_sigloader   #SigLoader,

# transform = None

parser = argparse.ArgumentParser(description='pre-train learning')

parser.add_argument('--lr', default=3e-2, type=float, help='Init learning rate')
# parser.add_argument('--train_num_perclass', default=1000, type=int, help='train_num_perclass')
# parser.add_argument('--test_num_perclass', default=400, type=int, help='test_num_perclass')
parser.add_argument('--n_epoch', default=300, type=int, help='Number of epochs')
parser.add_argument('--save_epoch', default=100, type=int, help='Number of  save epochs')
parser.add_argument('--class_num', default=20, type=int, help='all number of classes')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--test_batch_size', default=250, type=int, help='test batch size')
parser.add_argument('--log_step', default=100, type=int, help='log_step')
parser.add_argument('--drop_out', default=0.1, type=float, help='drop_out of network')
parser.add_argument('--sp_len', default=4800, type=int, help='sp_len')
parser.add_argument('--sample_len', default=4800, type=int, help='sample_len')
parser.add_argument('--data_root', default=r'E:\文献阅读\增量学习\徐云公开数据集', type=str, help='root of the data')
args = parser.parse_args()

repeatn = 1
# data_root = r'D:\学习\自监督学习\对比\师兄代码 - 副本\data.h5'
device = torch.device('cuda:0')
pre_path = 'initial_20class_普通卷积'
path = 'Dataset_50'
save_results = 'results'
save_feats = 'feats'
save_net = 'net'
# save_path1 = r'E:\文献阅读\增量学习\增量学习代码3\batch_loss'
# save_path2 = r'E:\文献阅读\增量学习\增量学习代码3\epoch_loss'
model_root = 'weights'

def test(dataloader,my_net,class_num):
    test_iter = iter(dataloader)
    i = 0
    n_test_correct = 0
    loss_test_sum = 0.
    conf = np.zeros([class_num, class_num])
    feats = []
    labels = []
    # my_net = model
    while i < len(dataloader):

        my_net.eval()

        img, label = next(test_iter)
        img, label = img.float().to(device), label.to(device)

        with torch.no_grad():
            class_output,feat_output = my_net(img)
            err_label = loss_class(class_output, label.long())
            feats.extend(feat_output.tolist())
            labels.extend(label.tolist())
            pred = class_output.data.max(1, keepdim=True)[1]
            n_test_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            loss_test_sum += float(err_label)
            for j in range(len(pred)):
                conf[pred.data.cpu().numpy()[j], label.data.cpu().numpy()[j]] += 1.0

        i += 1

    return float(n_test_correct)/float(len(dataloader.dataset)), loss_test_sum/float(len(dataloader)), conf,feats,labels


if __name__ == '__main__':
	for rpi in range(repeatn):

		itransform = transforms.Compose([
			# augmentations.RandomExchange(0.5),
			# augmentations.RandomReverse(0.5),
			# augmentations.PhaseShift(1.),  # 相位翻转
			# augmentations.Jitter(0.01),
			# augmentations.FFT(),
			# augmentations.RandomErase(1., [0., 0.5]),
			# augmentations.PhaseShift(1.),
			# augmentations.RandomEraseScatter(1., [0., 0.5]),
			# augmentations.RandomCrop(3000),  # 随机裁剪
			# augmentations.Permutation(2),
			# augmentations.Scaling(0.03),
			# augmentations.MagnitudeWarp(0.1, 256),
			# augmentations.JitterWarp(0.03, 64),
			# augmentations.SegRandomScale(0.03, 256),
			# augmentations.IFFT(),
			augmentations.Normalize(),
			# augmentations.Jitter2(0.5),     #噪声
		])
		test_transform = transforms.Compose([
			augmentations.Normalize(),
			# augmentations.Jitter2(0.2),
		])



		# class_num = 8
		dataset_train, dataset_test = get_sigloader(args.data_root, args.class_num, args.sp_len,args.sample_len, itransform,test_transform)

		dataloader_train = torch.utils.data.DataLoader(
			dataset=dataset_train,
			batch_size=args.batch_size,
			shuffle=True,
			drop_last=True,  #
			num_workers=5,
			# pin_memory = True
		)

		dataloader_test = torch.utils.data.DataLoader(
			dataset=dataset_test,
			batch_size=args.test_batch_size,
			shuffle=True,
			drop_last=False,
			num_workers=5,
			# pin_memory = True
		)

		my_net = resnet.resnet18(num_classes=args.class_num, drop_out=args.drop_out).to(device)
		print(my_net)
		# trainable_num = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
		# print('number of trainable parameters:', trainable_num)

		optimizer = optim.Adam(my_net.parameters(), lr=args.lr)
		scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
		acc_train = np.zeros([args.n_epoch, ])
		loss_train = np.zeros([args.n_epoch, ])
		acc_test = np.zeros([args.n_epoch, ])
		loss_test = np.zeros([args.n_epoch, ])
		conf_test = np.zeros([args.class_num, args.class_num])
		#
		if not os.path.exists(pre_path):
			os.mkdir(pre_path)
		if not os.path.exists(pre_path+ '/' + path):
			os.mkdir(pre_path+ '/' + path)
		if not os.path.exists(pre_path+ '/' + path + '/' + save_results):
			os.mkdir(pre_path+ '/' + path + '/' + save_results)
		if not os.path.exists(pre_path+ '/' + path + '/' + save_feats):
			os.mkdir(pre_path+ '/' + path + '/' + save_feats)
		if not os.path.exists(pre_path+ '/' + path + '/' + save_net):
			os.mkdir(pre_path+ '/' + path + '/' + save_net)
		# if not os.path.exists(str(rpi)+'/'+model_root):
		#     os.mkdir(str(rpi)+'/'+model_root)

		for epoch in range(args.n_epoch):

			train_iter = iter(dataloader_train)

			n_train_correct = 0
			loss_train_sum = 0.

			i = 0
			while i < len(dataloader_train):
				my_net.train()
				#
				img, label = next(train_iter)
				img, label = img.float().to(device), label.to(device)

				# batch_size = len(label)
				loss_class = nn.CrossEntropyLoss()
				class_output, _ = my_net(img)
				err_label = loss_class(class_output, label.long())

				optimizer.zero_grad()
				err_label.backward()
				optimizer.step()

				pred = class_output.data.max(1, keepdim=True)[1]
				n_train_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
				loss_train_sum += float(err_label)

				i += 1
				if (i % args.log_step == 0):
					print('Epoch [{}/{}] Step [{}/{}]: 原始模型 loss class={:.5f}'.format(
						epoch, args.n_epoch, i, len(dataloader_train), err_label))

			scheduler.step()
			acc_train[epoch] = float(n_train_correct) / float(len(dataset_train))
			loss_train[epoch] = loss_train_sum / float(len(dataloader_train))

			acc_test[epoch], loss_test[epoch], conf_test, feats_test, labels_test = test(dataloader_test, my_net,
																						 args.class_num)

			np.savetxt(pre_path+ '/' + path + '/' + save_results + '/{}_acc_train.txt'.format('初始模型'), acc_train)
			np.savetxt(pre_path+ '/' + path + '/' + save_results + '/{}_loss_train.txt'.format('初始模型'), loss_train)
			np.savetxt(pre_path+ '/' + path + '/' + save_results + '/{}_acc_test.txt'.format('初始模型'), acc_test)
			np.savetxt(pre_path+ '/' + path + '/' + save_results + '/{}_loss_test.txt'.format('初始模型'), loss_test)
			np.savetxt(pre_path+ '/' + path + '/' + save_results + '/{}_conf_test_epoch'.format('初始模型') + str(epoch) + '.txt',
					   conf_test)

			if ((epoch + 1) % args.save_epoch == 0):
				# torch.save(my_net, '{0}/model_epoch_{1}.pth'.format(str(rpi)+'/'+model_root, epoch))
				np.savetxt(pre_path+ '/' + path + '/' + save_feats + '/{0}_test_feat0_{1}.txt'.format('初始模型', epoch),
						   feats_test)
				np.savetxt(pre_path+ '/' + path + '/' + save_feats + '/{0}_test_label0_{1}.txt'.format('初始模型', epoch),
						   labels_test)

			print('初始模型 epoch: {} train acc: {:.4f} train loss: {:.4f} test acc: {:.4f} test loss: {:.4f}'.format(
				epoch,
				acc_train[epoch], loss_train[epoch], acc_test[epoch], loss_test[epoch]))
		# torch.save(my_net, pre_path + '/' + save_net + '/my_net.pkl')

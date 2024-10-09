# import sys
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
from data_loader import get_sigloader2  #get_sigloader2, get_sigloader1 #SigLoader,
# from torchvision import datasets
from torchvision import transforms
# import resnet
import numpy as np
# from collections import OrderedDict
import increment_model
from loss import MultiClassCrossEntropy as loss_old
from loss import loss_function2 as loss_new
import argparse
import copy
import augmentations
# import increment_model as im

device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='self-supervised learning')

parser.add_argument('--lr', default=1e-5, type=float, help='Init learning rate')
parser.add_argument('--K', default=1000, type=int, help='Negative samples queue length')
parser.add_argument('--τ', default=5, type=float, help='Temperature hyper-parameter of self-supervised learning')
parser.add_argument('--T', default=20, type=float, help='Temperature hyper-parameter of increment learning')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum coefficient')
parser.add_argument('--n_epoch', default=2000, type=int, help='Number of epochs')
parser.add_argument('--save_epoch', default=100, type=int, help='Number of  save epochs')
parser.add_argument('--class_num', default=10, type=int, help='total classes')
parser.add_argument('--new_classes', default=10, type=int, help='increment classes')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--test_batch_size', default=250, type=int, help='test batch size')
parser.add_argument('--log_step', default=100, type=int, help='log_step')
parser.add_argument('--drop_out', default=0.1, type=float, help='drop_out of network')
parser.add_argument('--sp_len', default=4800, type=int, help='sp_len')
parser.add_argument('--sample_len', default=4800, type=int, help='sample_len')
parser.add_argument('--data_root', default=r'E:\文献阅读\增量学习\徐云公开数据集', type=str, help='root of the data')
parser.add_argument('--net_root', default=r'E:\文献阅读\增量学习\徐云公开数据集\initial\net\my_net.pkl', type=str, help='root of the init net')
args = parser.parse_args()

pre_path = 'self_supervised_learning'
save_results = 'results'
# save_feats = 'feats'
save_net = 'net'
# save_path1 = r'E:\文献阅读\增量学习\增量学习代码\batch_loss'
# save_path2 = r'E:\文献阅读\增量学习\增量学习代码\epoch_loss'
# model_root = 'weights'
cudnn.benchmark = True
repeatn = 1
lamda = 0.5  #增量学习损失函数占的比重


if __name__ == '__main__':



    for rpi in range(repeatn):

        itransform = transforms.Compose([
            # augmentations.PhaseShift(1.),         #相位翻转
            # augmentations.Jitter(0.03),  #0.03
            # augmentations.Jitter2(0.03),
            augmentations.RandomCrop(4096),   #随机裁剪
            augmentations.Normalize(),
        ])

        dataset_train = get_sigloader2(args.data_root, args.sp_len,  args.sample_len, itransform)


        dataloader_train = torch.utils.data.DataLoader(
            dataset = dataset_train,
            batch_size = args.batch_size,
            shuffle = True,
            drop_last = True,             #
            num_workers = 5,
            # pin_memory = True
        )


        resnet = torch.load(args.net_root)
        # print(resnetq)
        resnetq = increment_model.increment_classes(resnet,args.new_classes)
        print(resnetq)

        resnetk = copy.deepcopy(resnetq)
        resnetq.to(device)
        resnetk.to(device)
        optimizer = optim.Adam(resnetq.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.8)
        loss_train = np.zeros([args.n_epoch, ])
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        if not os.path.exists(pre_path + '/' + save_results):
            os.mkdir(pre_path + '/' + save_results)
        if not os.path.exists(pre_path + '/' + save_net):
            os.mkdir(pre_path + '/' + save_net)
        KK = random.randint(1, args.K)
        dataloader_queue = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=KK,
            shuffle=True,
            drop_last=True,  #
            num_workers=5,
            # pin_memory = True
        )
        queue_iter = iter(dataloader_queue)
        queue_img, _, _ = next(queue_iter)
        img = queue_img.float().to(device)
        # resnetk(img).grad.zero_()
        queue,_ = resnetk(img)
        queue = queue.detach()
        for epoch in range(args.n_epoch):

            train_iter = iter(dataloader_train)

            n_train_correct = 0
            loss_train_sum = 0.

            i = 0
            while i < len(dataloader_train):
                #
                resnetq.train()
                imgq, imgk, _ = next(train_iter)

                xq = imgq.float().to(device)
                q,_ = resnetq(xq)
                xk = imgk.float().to(device)
                k,_ = resnetk(xk)
                k_increment,_ = resnetk(xq)
                k = k.detach()
                k_increment = k_increment.detach()
                loss2 = loss_old(q[:,:-args.new_classes],k_increment[:,:-args.new_classes],args.T)
                # loss2 = loss_old(q[:, :], k_increment[:, :], args.T)
                # 将输出规范化，使它们成为单位向量
                q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
                k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

                loss1 = loss_new(q, k, queue, args.τ)
                # loss = (1-lamda) * loss1 + lamda * loss2
                # print('loss_self',loss1)
                # print('loss_increment',loss2)
                loss = (1-lamda) * loss1 + lamda * loss2
                # print('loss=',  float(loss))
                # loss.requires_grad_(True)
                loss.backward()
                # print('loss=', float(loss))
                # 运行优化器
                optimizer.step()
                queue = torch.cat((queue, k), 0)

                if queue.shape[0] > args.K:
                    queue = queue[args.batch_size:, :]
                for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
                    θ_k.data.copy_(args.momentum * θ_k.data + θ_q.data * (1.0 - args.momentum))
                loss_train_sum += float(loss)
                i += 1
            scheduler.step()
            print('Epoch {} :LOSS {}'.format(epoch, loss))
            loss_train[epoch] = loss_train_sum / float(len(dataloader_train))
        if loss <= 0.2 and epoch >= 300 and k_num <= 4:
            torch.save(resnetq, pre_path + '/' + 'temporary' + '/' + str(k_num) + '/resnetq.pkl')
            print('write dowm')
            k_num = k_num + 1

        np.savetxt(pre_path + '/' + save_results + '/loss_train.txt', loss_train)
        # loss_epoch.append(loss.cpu().numpy())
    # np.savetxt(save_path2 + '.txt', loss_epoch)
    torch.save(resnetq, pre_path + '/' + save_net + '/resnetq.pkl')
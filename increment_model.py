import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import time
import copy
import torch.optim as optim
import augmentations
from torchvision import transforms
# from data_loader0 import get_sigloader
import os
from loss import MultiClassCrossEntropy as loss_old
# from loss import KDLoss as loss_old
# from data_loader0 import get_sigloader1

def test(dataloader,my_net,class_num):
    device = torch.device('cuda:0')
    test_iter = iter(dataloader)
    loss_class = nn.CrossEntropyLoss()
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

def kaiming_normal_init(m):
	#############if isinstance(m, nn.Conv2d):
	if isinstance(m, nn.Conv1d):  #判断是否是同一类型
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

def increment_classes(model,new_classes):
    prev_model = copy.deepcopy(model)
    prev_model.cuda()
    in_features = model.fc.in_features
    out_features = model.fc.out_features
    weight = model.fc.weight.data
    new_out_features = out_features + new_classes
    model.fc = nn.Linear(in_features, new_out_features, bias=False)
    # fc = model.fc
    # model.fc = nn.Linear(in_features, new_out_features, bias=False)
    # model.fc = model.fc
    kaiming_normal_init(model.fc.weight)
    model.fc.weight.data[:out_features] = weight
    return model,prev_model

def update(prev_model, model, dataset_train, train_batch_size, dataset_test_new, dataset_test_old, test_batch_size, n_epoch, new_classes, total_classes, lamda=0.5):
    device = torch.device('cuda:0')
    save_feats = 'feat_increment'
    save_results = 'results_increment'
    model_root = 'model_increment'
    rpi = 'total'
    # rpii = '0'
    # prev_model2 = copy.deepcopy(prev_model)
    # prev_model2.cuda()

    # itransform = transforms.Compose([
    #     augmentations.PhaseShift(1.),  # 相位翻转
    #     augmentations.Jitter(0.03),
    #     augmentations.RandomCrop(4096),  # 随机裁剪
    #     augmentations.Normalize(),
    # ])


    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,  #
        num_workers=5,
        # pin_memory = True
    )

    dataloader_test_new = torch.utils.data.DataLoader(
        dataset=dataset_test_new,
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=5,
        # pin_memory = True
    )
    dataloader_test_old = torch.utils.data.DataLoader(
        dataset=dataset_test_old,
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=5,
        # pin_memory = True
    )
    # dataloader_test_total = torch.utils.data.DataLoader(
    #     dataset= dataset_test_total,
    #     batch_size=test_batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     # pin_memory = True
    # )

    my_net = model.to(device)
    optimizer = optim.Adam(my_net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    acc_train = np.zeros([n_epoch, ])
    loss_train = np.zeros([n_epoch, ])
    acc_test_new = np.zeros([n_epoch, ])
    loss_test_new = np.zeros([n_epoch, ])
    acc_test_old = np.zeros([n_epoch, ])
    loss_test_old = np.zeros([n_epoch, ])
    # acc_test_total = np.zeros([n_epoch, ])
    # loss_test_total = np.zeros([n_epoch, ])

    if not os.path.exists(str(rpi)):
        os.mkdir(str(rpi))
    if not os.path.exists(str(rpi) + '/' + save_feats):
        os.mkdir(str(rpi) + '/' + save_feats)
        if not os.path.exists(str(rpi) + '/' + save_results):
            os.mkdir(str(rpi) + '/' + save_results)
    if not os.path.exists(str(rpi) + '/' + model_root):
        os.mkdir(str(rpi) + '/' + model_root)

    for epoch in range(n_epoch):
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
            loss_new = nn.CrossEntropyLoss()
            # loss_old = MultiClassCrossEntropy()
            new_net_output, _ = my_net(img)
            old_net_output,_ = prev_model(img)
            new_loss = loss_new(new_net_output, label.long())
            # print("size 查看1",new_net_output[:,:-new_classes].size())
            # print("size 查看2", old_net_output.size())
            # old_loss = loss_old(new_net_output, old_net_output)
            old_loss = loss_old(new_net_output[:, :-new_classes], old_net_output)

            total_loss = (1-lamda) * new_loss + lamda * old_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # prev_model = momentum_net.momentum(prev_model,my_net)
            # for θ_k, θ_q in zip(prev_model.parameters(), my_net.parameters()):
            #     θ_k.data.copy_(0.99 * θ_k.data + θ_q.data * (1.0 - 0.99))

            pred = new_net_output.data.max(1, keepdim=True)[1]
            n_train_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            loss_train_sum += float(total_loss)
            i += 1
            # if (i % 10 == 0):
            #     print('增量学习过程中 Epoch [{}/{}] Step [{}/{}]:  loss class={:.5f}'.format(
            #         epoch, n_epoch, i, len(dataloader_train), total_loss))
        scheduler.step()
        acc_train[epoch] = float(n_train_correct) / float(len(dataset_train))
        loss_train[epoch] = loss_train_sum / float(len(dataloader_train))

        acc_test_new[epoch], loss_test_new[epoch], conf_new, feats_new, labels_new = test(dataloader_test_new, my_net, total_classes)
        acc_test_old[epoch], loss_test_old[epoch], conf_old, feats_old, labels_old = test(dataloader_test_old, my_net, total_classes)
        # acc_test_total[epoch], loss_test_total[epoch], conf_test_total, feats_test_total, labels_test_total = test(total_dataset_test, my_net, total_classes)

        np.savetxt(str(rpi) + '/' + save_results + '/acc_train.txt', acc_train)
        np.savetxt(str(rpi) + '/' + save_results + '/loss_train.txt', loss_train)
        np.savetxt(str(rpi) + '/' + save_results + '/acc_test_new.txt', acc_test_new)
        np.savetxt(str(rpi) + '/' + save_results + '/loss_test_new.txt', loss_test_new)
        np.savetxt(str(rpi) + '/' + save_results + '/acc_test_old.txt', acc_test_old)
        np.savetxt(str(rpi) + '/' + save_results + '/loss_test_old.txt', loss_test_old)
        np.savetxt(str(rpi) + '/' + save_results + '/{}_conf_test_old'.format('旧任务') + str(epoch) + '.txt',
                   conf_old)
        np.savetxt(str(rpi) + '/' + save_results + '/{}_conf_test_total'.format('所有任务') + str(epoch) + '.txt',
                   conf_new)
        # # np.savetxt(str(rpi) + '/' + save_path + '/{}_acc_test.txt'.format('所有任务'), acc_test_total)
        # # np.savetxt(str(rpi) + '/' + save_path + '/{}_loss_test.txt'.format('所有任务'), loss_test_total)
        # np.savetxt(str(rpi) + '/' + save_path + '/{}_acc_test.txt'.format('旧任务'), acc_test_total)
        # np.savetxt(str(rpi) + '/' + save_path + '/{}_loss_test.txt'.format('旧任务'), loss_test_total)

        if ((epoch + 1) % 40 == 0):
            torch.save(my_net, '{0}/model_epoch_{1}.pth'.format(str(rpi)+'/'+model_root, epoch))
            # np.savetxt(str(rpi) + '/' + save_path2 +'/{0}_test_feat0_{1}.txt'.format('所有任务', epoch), feats_test_total)
            # np.savetxt(str(rpi) + '/' + save_path2 +'/{0}_test_label0_{1}.txt'.format('所有任务', epoch), labels_test_total)
            np.savetxt(str(rpi) + '/' + save_feats + '/{0}_test_feat_new_{1}.txt'.format('新任务', epoch), feats_new)
            np.savetxt(str(rpi) + '/' + save_feats + '/{0}_test_label_new_{1}.txt'.format('新任务', epoch), labels_new)
            np.savetxt(str(rpi) + '/' + save_feats + '/{0}_test_feat_old_{1}.txt'.format('旧任务', epoch), feats_old)
            np.savetxt(str(rpi) + '/' + save_feats + '/{0}_test_label_old_{1}.txt'.format('旧任务', epoch), labels_old)
        print('增量学习模型 epoch: {} 训练准确率: {:.4f} 训练损失: {:.4f} 新任务测试准确率: {:.4f} 新任务测试损失: {:.4f} 旧任务测试准确率: {:.4f} 旧任务测试损失: {:.4f}'.format(epoch,
              acc_train[epoch],loss_train[epoch],acc_test_new[epoch],loss_test_new[epoch],acc_test_old[epoch],loss_test_old[epoch]))
        # print('总任务 epoch: {} test acc: {:.4f} test loss: {:.4f}'.format(
        #         epoch,acc_test_total[epoch], loss_test_total[epoch]))
        # print('旧任务 epoch: {} test acc: {:.4f} test loss: {:.4f}'.format(
        #     epoch, acc_test_total[epoch], loss_test_total[epoch]))
        # if acc_test_new[epoch] >= 0.9:
        #     break


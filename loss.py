import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F

def MultiClassCrossEntropy(logits, labels, T=2):#L_old的计算
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).cuda()


def KDLoss(preds, gts,T=2):
	LS = nn.LogSoftmax(dim=-1)
	preds = F.softmax(preds, dim=-1)
	preds = torch.pow(preds, 1. / T)
	# l_preds = F.softmax(preds, dim=-1)
	l_preds = LS(preds)
	gts = F.softmax(gts, dim=-1)
	gts = torch.pow(gts, 1. / T)
	l_gts = LS(gts)
	l_preds = torch.log(l_preds)
	l_preds[l_preds != l_preds] = 0.  # Eliminate NaN values
	loss = torch.mean(torch.sum(-l_gts * l_preds, axis=1))
	return loss


def loss_function(q, k, queue,τ = 0.05):

    # N是批量大小
    N = q.shape[0]

    # C是表示的维数
    C = q.shape[1]

    # bmm代表批处理矩阵乘法
    # 如果mat1是b×n×m张量，那么mat2是b×m×p张量，
    # 然后输出一个b×n×p张量。
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))

    # 在查询和队列张量之间执行矩阵乘法
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),τ)), dim=1)

    # 求和
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))


def loss_function2(q, k, queue,τ = 0.05):

    # N是批量大小
	device = torch.device('cuda:0')
	N = q.shape[0]

    # C是表示的维数
	C = q.shape[1]
	loss_class = nn.CrossEntropyLoss()

	K = queue.shape[0]
	# print(queue.shape)

    # bmm代表批处理矩阵乘法
    # 如果mat1是b×n×m张量，那么mat2是b×m×p张量，
    # 然后输出一个b×n×p张量。
	l_pos = torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1)
	l_neg = torch.mm(q.view(N, C), queue.view(C, K))
	# print('l_pos.shape',l_pos.shape)
	# print('l_neg.shape',l_neg.shape)
	logits = torch.cat([l_pos,l_neg], dim = 1)
	labels = torch.zeros(N).to(device)
	labels = labels.long()
	# print('labels.shape',labels.shape)
	loss2 = loss_class(logits/τ, labels)
	# print('loss2',loss2)
	return loss2

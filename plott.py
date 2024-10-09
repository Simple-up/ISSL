from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
# filename1 = r'E:\文献阅读\增量学习\增量学习代码3\initial\results\初始模型_acc_test.txt'
# filename2 = r'E:\文献阅读\增量学习\增量学习代码3\initial\results\初始模型_acc_train.txt'
# filename3 = r'E:\文献阅读\增量学习\徐云公开数据集\self_supervised_learning\results\loss_train.txt'
# filename4 = r'E:\文献阅读\增量学习\增量学习代码3\Transfer_learning\93%-400新-100旧\results\迁移过程_acc_train.txt'
# filename1 = r'E:\文献阅读\增量学习\增量学习代码3\Transfer_learning\results\迁移过程_acc_test.txt'
# filename2 = r'E:\文献阅读\增量学习\增量学习代码3\Transfer_learning\results\迁移过程_acc_train.txt'
# filename1 = r'E:\文献阅读\增量学习\增量学习代码3\comparison\results\对比过程_acc_test.txt'
filename2 = r'E:\文献阅读\增量学习\徐云公开数据集\self_supervised_learning\两个LOSS\loss_train_7.txt'
filename1 = r'E:\文献阅读\增量学习\徐云公开数据集\self_supervised_learning\两个LOSS\loss_train_9.txt'

with open(filename1, 'r') as ff:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
    # 然后将每个元素中的不同信息提取出来
    acc_text = ff.read().splitlines()
acc_text1 = list(map(float, acc_text))
acc_text1 = [float(x)-1.0 for x in acc_text1]
print(acc_text1)
with open(filename2, 'r') as ff:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
    # 然后将每个元素中的不同信息提取出来
    acc_text2 = ff.read().splitlines()
acc_text2 = list(map(float, acc_text2))

# print(acc_text2)



# with open(filename3, 'r') as ff:
#     # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
#     # 然后将每个元素中的不同信息提取出来
#     acc_text_3 = ff.read().splitlines()
# acc_text3 = list(map(float, acc_text_3))
# maxZ = max(acc_text1)
# print('初始模型_acc_test',maxZ)
# k = 0
# for i in acc_text1:
#     if i == maxZ:
#         print("序列号",k+1)
#         print(acc_text1[k])
#     k = k + 1
# with open(filename2, 'r') as ff:
#     acc_train = ff.read().splitlines()
# acc_train1 = list(map(float, acc_train))
# # print('初始模型_acc_train',acc_train1[-1])
x = np.linspace(0,len(acc_text1)-1,len(acc_text1))
x1 = np.linspace(0,len(acc_text2)-1,len(acc_text2))
# x2 = np.linspace(0,len(acc_text3)-1,len(acc_text3))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# with open(filename3, 'r') as ff:
#     # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
#     # 然后将每个元素中的不同信息提取出来
#     acc_text2 = ff.read().splitlines()
# acc_text2 = list(map(float, acc_text2))
# maxZ = max(acc_text1)
# print('初始模型_acc_test',maxZ)
# k = 0
# for i in acc_text1:
#     if i == maxZ:
#         print("序列号",k+1)
#         print(acc_text1[k])
#     k = k + 1
# with open(filename4, 'r') as ff:
#     acc_train2 = ff.read().splitlines()
# acc_train2 = list(map(float, acc_train2))
# # print('初始模型_acc_train',acc_train1[-1])
# y = np.linspace(0,len(acc_text2)-1,len(acc_text2))


# with open(filename3, 'r') as ff:
#     # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
#     # 然后将每个元素中的不同信息提取出来
#     acc_text = ff.read().splitlines()
# acc_text2 = list(map(float, acc_text))
# print('增量训练过程_acc_test',max(acc_text2))
# with open(filename4, 'r') as ff:
#     # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
#     # 然后将每个元素中的不同信息提取出来
#     acc_train = ff.read().splitlines()
# acc_train2 = list(map(float, acc_train))
# print('增量训练过程_acc_train',max(acc_train2))
# print('提升量为{}%'.format((acc_text2[-1]-acc_text1[-1])*100))
# with open(filename5, 'r') as ff:
#     # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
#     # 然后将每个元素中的不同信息提取出来
#     acc_text = ff.read().splitlines()
# acc_text3 = list(map(float, acc_text))
# print('旧任务最大值_acc_text',max(acc_text3))
# print('旧任务最小值_acc_text',min(acc_text3))
# y = np.linspace(0,len(acc_text2)-1,len(acc_text2))

plt.plot(x,acc_text1,label='-MU')
plt.plot(x1,acc_text2, 'r',label='+MU')
# plt.plot(x2,acc_text3,label='训练损失目前结果')
# # plt.plot(x,acc_text1, '#027BC6',label='测试准确率')
# # plt.plot(x,acc_train1,'r', label='训练准确率')#, label='0.8728'
plt.xlabel('Epoch',fontsize=16,fontname='Times New Roman')
# # plt.xlabel('初始模型Epoch',fontsize=16)
# plt.ylabel('Loss',fontsize=10)
plt.ylabel('Loss',fontsize=16,fontname='Times New Roman')
plt.legend()
plt.show()


# fig = plt.figure()
# ax1 = fig.add_subplot(1, 2, 1)
# plt.annotate(f'准确率:{int(acc_text1[-1]*100)}%', xy=(x[-1], acc_text1[-1]), xytext=(x[-1]-1, acc_text1[-1]+0.11),
#              arrowprops=dict(facecolor='red'))
#
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.plot(x,acc_text1, 'mediumseagreen',label='测试准确率')
# ax1.plot(x,acc_train1,'r', label='训练准确率')
# ax2.plot(y,acc_text2, 'mediumseagreen',label='所有目标的测试准确率')
# ax2.plot(y,acc_train2,'r', label='训练准确率')
# ax1.legend()
# ax1.set_title('原始模型训练')
# ax1.set_xlabel('初始模型Epoch',fontsize=11)
# ax1.set_ylabel('初始模型准确率%',fontsize=11)
# ax2.legend()
# ax2.set_title('下游任务过程训练')
# ax2.set_xlabel('Epoch',fontsize=11)
# ax2.set_ylabel('所有任务准确率%',fontsize=11)
# plt.annotate(f'准确率:{int(acc_text2[-1]*100)}%', xy=(y[-1], acc_text2[-1]), xytext=(y[-1]-1, acc_text2[-1]+0.15),
#              arrowprops=dict(facecolor='red'))
# plt.show()
# plt.subplot(1,2,1)
# plt.suptitle("原始模型训练",fontsize=10)
# plt.plot(x,acc_text1, '#027BC6',label='测试准确率')
# plt.plot(x,acc_train1,'r', label='训练准确率')#, label='0.8728'
# plt.xlabel('初始模型Epoch',fontsize=10)
# # plt.xlabel('初始模型Epoch',fontsize=16)
# plt.ylabel('初始模型准确率%',fontsize=10)
# plt.legend()
# plt.subplot(1,2,2)
# plt.suptitle("增量学习过程训练",fontsize=10)
# plt.plot(y,acc_text2, '#027BC6',label='新增样本测试准确率')
# plt.plot(y,acc_train2,'r', label='训练准确率')#, label='0.8728'
# # plt.plot(y,acc_text3, 'SeaGreen',label='总体准确率')
# plt.xlabel('增量学习Epoch',fontsize=10)
# plt.ylabel('增量学习准确率%',fontsize=10)
# # plt.xlabel('增量学习Epoch',fontsize=16)
# # plt.ylabel('增量学习准确率%',fontsize=16)
# plt.legend()

# plt.show()

'''
原始模型_acc_test 0.8521875
原始模型_acc_train 0.8966666666666666
增量训练过程_acc_test 0.90875
增量训练过程_acc_train 0.9722222222222222
提升量为5.500000000000005%


原始模型_acc_test 0.8465625
原始模型_acc_train 0.8482142857142857
增量训练过程_acc_test 0.9146875
增量训练过程_acc_train 0.94875
提升量为6.218749999999995%
'''
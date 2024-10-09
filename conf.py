import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#999 1-1,899 2-2,799 3-3
# conf1 = np.array(np.loadtxt(r'D:\学习\自监督学习\对比\师兄代码 - 副本\0\2020.9.13（特征图前三章）\results\conf_test_epoch799.txt',dtype="float"))
conf1 = np.array(np.loadtxt(r'E:\文献阅读\增量学习\徐云公开数据集\Transfer_learning\results_25\迁移过程_conf_test_epoch299.txt',dtype="int"))
# 使用列表推导式对数组中的每个元素进行格式化

conf = ["{:.0f}".format(x) for x in conf1.flatten()]
# np.set_printoptions(precision=1, suppress=True)
# 如果需要将格式化后的数组转换回numpy数组
# conf1 = np.array(conf1_formatted)
print(conf1.shape)
# x_tick=['1','2','3','4','5','6','7','8']
# ['0','1','2','3','4','5','6','7']
# y_tick=['1','2','3','4','5','6','7','8']
# # X=[[1,2,3],[4,5,6],[7,8,9]]
# data={}
# for i in range(8): data[x_tick[i]] = conf1[i]
# pd_data=pd.DataFrame(data,index=y_tick,columns=y_tick)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure()
ax = sns.heatmap(conf1,annot=True,fmt='g')

plt.xlabel('Authentic labels',fontsize=16,fontname='Times New Roman')
plt.ylabel('Predictive labels',fontsize=16,fontname='Times New Roman')
plt.xticks(ticks = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5],labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], fontsize=11) #x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(ticks = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5],labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], fontsize=11) #y轴刻度的字体大小（文本包含在pd_data中了）
plt.title('Confusion Matrix',fontsize=18,fontname='Times New Roman')
plt.show()
# plt.show()
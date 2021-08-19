# -*- coding: utf-8 -*-
"""
GaoMing
"""
#导入模块，如果导入失败的话，可能是你的电脑没有安装相应模块，可以使用pip install命令安装
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#导入数据，需要把数据和代码放在同一文件下或使用绝对引用
wine = pd.read_csv('data_new3.csv',index_col=0)

color = ['Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','Oranges','Oranges_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','cool_r','coolwarm','mako','mako_r', 'pink','pink_r','vlag','vlag_r','winter','winter_r']

# 无量纲化_极差变换法
def Dim(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()  #获取指标列表
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame

# 计算关联系数
def GRA_ONE(gray1, m=0 ,r=0.5):
    gray = gray1.copy()
    std = gray.iloc[:, m]  # 为标准要素
    gray.drop(gray.columns[m],axis=1,inplace=True) #删除标准列
    ce = gray.iloc[:, 0:]  # 为比较要素
    shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(ce.iloc[j, i] - std.iloc[j])

    # 取出矩阵中最大值与最小值
    c, d = a.max(), a.min()

    # 计算值
    result = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (d + r * c) / (a[i, j] + r * c)

    # 求均值，得到灰色关联值,并返回
    result_list = [np.mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m,1)
    result0 = pd.DataFrame(result_list)
    result0.index = gray1.columns
    result0.columns = [gray1.columns[m]]
    return result0

def GRA(df):
    df_local = pd.DataFrame(index = df.columns, columns=df.columns)
    for i in range(len(df.columns)):
        GRA0 = GRA_ONE(df, m=i)
        df_local.loc[:,GRA0.columns[0]] = GRA0.iloc[:,0]
    return df_local

def ShowM(DataFrame,n):   #n = 0:28
    colormap = plt.cm.RdBu
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(14, 14))
    ax.set_title('GRA HeatMap')
    
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('dark'): #五种主体风格
        sns.heatmap(DataFrame,
                    cmap=color[n],
                    annot=True,
                    #mask=mask,
                   )
    plt.show()


wine1 = Dim(wine)
wine2 = GRA_ONE(wine1)
wine3 = GRA(wine1)
#ShowM的后一个参数是颜色参数，取值为0~27
ShowM(wine3,12)


wine1.iloc[:,[3,6]].plot()
wine1.iloc[:,0:3].plot()
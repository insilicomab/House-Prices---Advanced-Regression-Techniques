# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:23:45 2021

@author: m-lin
"""
"""
コメント：
baseline作成後のEDA
"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 学習データの読み込み
train = pd.read_csv('./data/train.csv')

# SalePriceの各統計量を確認
train['SalePrice'].describe()

# SalePriceのヒストグラムを描画
plt.hist(train['SalePrice'], bins=20)

# SalePriceを対数化
np.log(train['SalePrice'])

# 対数化したSalePriceをヒストグラムで可視化
plt.hist(np.log(train['SalePrice']), bins=20)

"""
コメント：
baseline ver2作成後
説明変数の前処理
"""

# テストデータの読み込み
test = pd.read_csv('./data/test.csv')

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# 欠損値の数が上位40の変数を確認
df.isnull().sum().sort_values(ascending=False).head(40) # ascending=False:降順

'''
PoolQC
'''

# PoolQCの各分類ごとの個数
df['PoolQC'].value_counts()

# PoolQCの値があるものを1、ないものを0に変換
df.loc[~df['PoolQC'].isnull(), 'PoolQC'] = 1
df.loc[df['PoolQC'].isnull(), 'PoolQC'] = 0

df['PoolQC'].value_counts()

'''
MiscFeature
'''

# MiscFeatureの各分類ごとの個数
df['MiscFeature'].value_counts()

# MiscFeatureの値があるものを1、ないものを0に変換
df.loc[~df['MiscFeature'].isnull(), 'MiscFeature'] = 1
df.loc[df['MiscFeature'].isnull(), 'MiscFeature'] = 0

df['MiscFeature'].value_counts()

'''
Alley
'''

# Alleyの各分類ごとの個数
df['Alley'].value_counts()

# MiscFeatureの値があるものを1、ないものを0に変換
df.loc[~df['Alley'].isnull(), 'Alley'] = 1
df.loc[df['Alley'].isnull(), 'Alley'] = 0

df['Alley'].value_counts()

# 高級住宅設備の数という特徴量を作成
df['hasHighFacility'] = df['PoolQC'] + df['MiscFeature'] + df['Alley']
df['hasHighFacility'] = df['hasHighFacility'].astype(int)
df['hasHighFacility'].value_counts()

# 元のデータからPoolQC、MiscFeature、Alleyを削除
df = df.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1)

"""
コメント：
baseline ver2作成後
外れ値の除外
"""

# 各統計量の確認
statisticDf = df.describe().T

# 数値データのみ抜き出し
train_num = train.select_dtypes(include=[np.number])

# 比例尺度ではない変数
nonratio_features = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond',
                     'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']

# 比例尺度データの抜き出し
# setに変換して差分を取った後、listに戻した上でsorted()で並びを固定
num_features = sorted(list(set(train_num)-set(nonratio_features))) 
num_features
train_num_rs = train_num[num_features]

# 第三四分位数が0となる変数を確認
for col in num_features:
    if train_num_rs.describe()[col]['75%'] == 0:
        print(col, len(train_num_rs[train_num_rs[col] == 0]))

# ある特定の値のみしか取らないものを確認
for col in num_features:
    if train_num_rs[col].nunique() < 15:
        print(col, train_num_rs[col].nunique())

# 外れ値（3SDの外側）があるか確認
for col in num_features:
    tmp_df = train_num_rs[(train_num_rs[col] > train_num_rs[col].mean() + train_num_rs[col].std()*3) |\
                          (train_num_rs[col] < train_num_rs[col].mean() - train_num_rs[col].std()*3)]
    print(col, len(tmp_df))

# BsmtFinSF1とSalePriceの分布を可視化
df.plot.scatter(x='BsmtFinSF1', y='SalePrice')

# BsmtFinSF1が広いもののSalePriceが高くないものを確認
df[df['BsmtFinSF1'] > 5000]

# TotalBsmtSFとSalePriceの分布を可視化
df.plot.scatter(x='TotalBsmtSF', y='SalePrice')
df[df['TotalBsmtSF']>6000]

# GrLivAreaとSalePriceの分布を可視化
df.plot.scatter(x='GrLivArea', y='SalePrice')
df[df['GrLivArea']>5000]

# 1stFlrSFとSalePriceの分布を可視化
df.plot.scatter(x='1stFlrSF', y='SalePrice')
df[df['1stFlrSF']>4000]


"""
コメント：
baseline ver3作成後
特徴量を生成する
"""

# 時間に関する変数の統計量を確認
df[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']].describe()

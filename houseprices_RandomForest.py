# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:01:43 2021

@author: m-lin
"""

'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

# データの確認
train.head()
train.dtypes

'''
特徴量エンジニアリング
'''

# ライブラリのインポート
from sklearn.preprocessing import LabelEncoder

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)


'''
コメント：
欠損値の多い高級住宅設備に関する変数をまとめる
'''
# PoolQCの値があるものを1、ないものを0に変換
df.loc[~df['PoolQC'].isnull(), 'PoolQC'] = 1
df.loc[df['PoolQC'].isnull(), 'PoolQC'] = 0

# MiscFeatureの値があるものを1、ないものを0に変換
df.loc[~df['MiscFeature'].isnull(), 'MiscFeature'] = 1
df.loc[df['MiscFeature'].isnull(), 'MiscFeature'] = 0

# MiscFeatureの値があるものを1、ないものを0に変換
df.loc[~df['Alley'].isnull(), 'Alley'] = 1
df.loc[df['Alley'].isnull(), 'Alley'] = 0

# 高級住宅設備の数という特徴量を作成
df['hasHighFacility'] = df['PoolQC'] + df['MiscFeature'] + df['Alley']
df['hasHighFacility'] = df['hasHighFacility'].astype(int)
df['hasHighFacility'].value_counts()

# 元のデータからPoolQC、MiscFeature、Alleyを削除
df = df.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1)

'''
コメント：
外れ値の除外
'''

# 外れ値以外を抽出（df['SalePrice'].isnull()とすることでテストデータはすべて抽出）
df = df[(df['BsmtFinSF1'] < 2000) | (df['SalePrice'].isnull())]
df = df[(df['TotalBsmtSF'] < 3000) | (df['SalePrice'].isnull())]
df = df[(df['GrLivArea'] < 4500) | (df['SalePrice'].isnull())]
df = df[(df['1stFlrSF'] < 2500) | (df['SalePrice'].isnull())]
df = df[(df['LotArea'] < 100000) | (df['SalePrice'].isnull())]

'''
コメント：
特徴量（築何年経過したか）を追加
'''

df['Age'] = df['YrSold'] - df['YearBuilt']

'''
コメント：
特徴量'TotalSF'（各階トータルの広さ）
特徴量'Total_Bathrooms'（お風呂の数の合計）
特徴量'Total_Porch'（Porchの広さの合計）
'''

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Total_Bathrooms'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']
df['Total_PorchSF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

'''
コメント：
Porchの広さの合計をPorchがあるかないかの0、1の値に変換
'''

df['hasPorch'] = df['Total_PorchSF'].apply(lambda x:1 if x>0 else 0)
df = df.drop('Total_PorchSF', axis=1)

'''
コメント：
LabelEncoderによるダミー変数化
'''

# object型の変数の取得
categories = df.columns[df.dtypes == 'object']
print(categories)

# 欠損値を数値に変換
for cat in categories:
    le = LabelEncoder() 
    print(cat)
    
    df[cat].fillna('missing', inplace=True)
    le = le.fit(df[cat])
    df[cat] = le.transform(df[cat])
    # LabelEncoderは数値に変換するだけであるため、最後にastype('category')としておく
    df[cat] = df[cat].astype('category') 

# 欠損値を各変数の中央値で補完
for col in df.columns:
    tmp_null_count = df[col].isnull().sum()
    if (tmp_null_count > 0) & (col != 'SalePrice'):
        print(col, tmp_null_count)
        df[col] = df[col].fillna(df[col].median())

# trainとtestに再分割
train = df[~df['SalePrice'].isnull()]
test = df[df['SalePrice'].isnull()]

'''
コメント：
目的変数SalePriceの対数化
'''

# SalePriceの対数化
train['SalePrice_log'] = np.log(train['SalePrice'])

# 説明変数と目的変数を指定
X_train = train.drop(['SalePrice', 'Id', 'SalePrice_log'], axis=1)
Y_train = train['SalePrice_log']
X_test = test.drop(['SalePrice', 'Id'], axis=1)

'''
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statistics import mean

# 3分割する
folds = 3
kf = KFold(n_splits=folds)

# 各foldごとに作成したモデルごとの予測値を保存
models = []
rmses = []
oof = np.zeros(len(X_train))

for train_index, val_index in kf.split(X_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]    
    
    model = rf(n_estimators=50,
               random_state=1234)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(rmse)
    
    models.append(model)
    rmses.append(rmse)
    oof[val_index] = y_pred

# 平均RMSEを計算する
mean(rmses)

# 現状の予測値と実際の値の違いを可視化
actual_pred_df = pd.DataFrame({
    'actual':Y_train,
    'pred': oof})

actual_pred_df.plot(figsize=(12,5))

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)

# predsの平均を計算
preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis = 0)

# 予測値を元のスケールに戻す
preds_exp = np.exp(preds_mean)
len(preds_exp)

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample_submission.csv')

# 'SalePrice'の値を置き換え
sub['SalePrice'] = preds_exp

# CSVファイルの出力
sub.to_csv('./submit/houseprices_rf.csv', index=False)

"""
予測精度：
0.1365992502831409
"""
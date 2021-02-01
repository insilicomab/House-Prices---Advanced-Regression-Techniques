# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:01:43 2021

@author: m-lin
"""

"""
コメント：
コメント：
ver2: EDA後にSalePriceを対数化し、正規分布化させ、予測精度の向上を確認する。
ver3: 説明変数の前処理、外れ値の除外
ver4: 特徴量を生成する
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

'''
モデルの構築と評価
'''

# ライブラリのインポート
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statistics import mean

# 3分割する
folds = 3
kf = KFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    'objective':'regression',
    'random_seed':1234    
}

# 各foldごとに作成したモデルごとの予測値を保存
models = []
rmses = []
oof = np.zeros(len(X_train))

for train_index, val_index in kf.split(X_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=100, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
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

# 特徴量重要度の表示
for model in models:
    lgb.plot_importance(model, importance_type='gain',
                        max_num_features=15)

"""
予測精度：
0.11735107013879016
"""
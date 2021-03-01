# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:01:43 2021

@author: m-lin
"""

"""
コメント：
Optunaでハイパーパラメータを最適化
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
    df[cat] = df[cat].astype('int8') 

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
Optunaでハイパーパラメータを最適化
'''

# ライブラリのインポート
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=0.2,
                                                      random_state=1234,
                                                      shuffle=False,
                                                      stratify=None)

def objective(trial):
    params = {
        "learning_rate":0.05,
        "seed":1234, 
        "max_depth":trial.suggest_int("max_depth",3,16),
        "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.2,0.9),
        "sublsample":trial.suggest_uniform("sublsample",0.2,0.9),
    }
    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgb_eval = xgb.DMatrix(x_valid, label=y_valid)
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]
    model = xgb.train(params, xgb_train,
                      evals=evals,
                      num_boost_round=1000,
                      early_stopping_rounds=20,
                      verbose_eval=10,)
    y_pred = model.predict(xgb_eval)
    score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=50)
print(study.best_params)

"""
best_params：
{'max_depth': 3,
 'colsample_bytree': 0.42079984564692874,
 'sublsample': 0.45459753965983585}
"""

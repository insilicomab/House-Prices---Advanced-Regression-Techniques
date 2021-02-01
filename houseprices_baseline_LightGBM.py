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

# 説明変数と目的変数を指定
X_train = train.drop(['SalePrice', 'Id'], axis=1)
Y_train = train['SalePrice']

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
    rmse = np.sqrt(mean_squared_error(np.log(y_valid), np.log(y_pred)))
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
0.1325997570084599
"""
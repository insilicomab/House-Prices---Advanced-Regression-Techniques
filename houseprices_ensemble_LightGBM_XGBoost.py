# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:07:15 2021

@author: m-lin
"""

'''
アンサンブル
'''

# ライブラリのインポート
import pandas as pd

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample_submission.csv')

# 予測データの読み込み
lgb_sub = pd.read_csv('./submit/houseprices_LightGBM.csv')
xgb_sub = pd.read_csv('./submit/houseprices_XGBoost.csv')

# アンサンブル（lgb*0.5 + xgb*0.5）
pred_ens = (lgb_sub['SalePrice'] + xgb_sub['SalePrice'])/2

# 'SalePrice'の値を置き換え
sub['SalePrice'] = pred_ens

# CSVファイルの出力
sub.to_csv('./submit/houseprices_ensemble.csv', index=False)

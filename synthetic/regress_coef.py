from evaluation import predictTastes
import torch
import pickle
import torch
from collections import OrderedDict
from data_utils import ChoiceDataset
from sklearn.linear_model import LinearRegression
import numpy as np


def regress(model, dic_z, dic_z_zall):
    pred_vots = []
    pred_vowts = []
    for group in dic_z:
        pred_tastes = predictTastes(model, dic_z[group])
        pred_vots.extend(pred_tastes['vots'].flatten().tolist())
        pred_vowts.extend(pred_tastes['vowts'].flatten().tolist())

    zall = torch.cat(list(dic_z_zall.values()), dim=0)

    reg_vot = LinearRegression().fit(zall, pred_vots)
    reg_vowt = LinearRegression().fit(zall, pred_vowts)

    coefs_time = [reg_vot.intercept_] + list(reg_vot.coef_)
    coefs_wait = [reg_vowt.intercept_] + list(reg_vowt.coef_)

    return coefs_time, coefs_wait
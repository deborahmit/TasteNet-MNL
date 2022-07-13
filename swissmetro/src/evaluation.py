import torch
from train import evaluate_epoch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from prettytable import PrettyTable


def predictChoice(model, dataset):
    with torch.no_grad():
        prob_choice, _ = model.forward(dataset.z, dataset.x_dict, dataset.av)
    return prob_choice, prob_choice.argmax(dim=1)


def summarizeDataset(model, ds):
    dl = DataLoader(ds, batch_size=model.args.batch_size, shuffle=False, num_workers=5)

    # nll
    nll = evaluate_epoch(model, dl)

    # predict choice 
    prob_choice, pred_choice = predictChoice(model, ds)
    
    # choice accuracy
    acc_choice = accuracy_score(ds.y, pred_choice)
    f1 = f1_score(ds.y, pred_choice, average='macro')
    sm = {"nll": nll, "acc": acc_choice, "f1": f1}
    return sm


def summarize(model, ds_train, ds_dev, ds_test):
    sm_train = summarizeDataset(model, ds_train)
    sm_dev = summarizeDataset(model, ds_dev)
    sm_test = summarizeDataset(model, ds_test)
    summary = {"train": sm_train, "dev": sm_dev, "test": sm_test}
    return summary


def printSummary(summary, precision=3):
    x = PrettyTable()
    data_list = ["train", "dev", "test"]
    x.field_names = [""] + ["acc_" + data for data in data_list] + \
                    ["nll_" + data for data in data_list] + \
                    ["f1_" + data for data in data_list]

    x.add_row(["model"] +
              [round(summary[data]["acc"], precision) for data in data_list] +
              [round(summary[data]["nll"], precision) for data in data_list] +
              [round(summary[data]["f1"], precision) for data in data_list])

    return x


def predictParams(model, z):
    '''
    Predict parameters based on input z
    '''
    with torch.no_grad():
        b = model.params_module(z)
        b = model.constraints(b)
    return b


def bToDataFrame(b):
    df = pd.DataFrame( \
        np.array([[b["TRAIN_TT"], b["SM_TT"], b["CAR_TT"]], \
                  [b["TRAIN_HE"], b["SM_HE"], 0.0], \
                  [0, b["SM_SEATS"], 0.0], \
                  [-1, -1, -1], \
                  [b["TRAIN_ONE"], b["SM_ONE"], 0.0]]), \
        index=["TT", "HE", "SEATS", "COST", "ONE"], \
        columns=["TRAIN", "SM", "CAR"])
    return df

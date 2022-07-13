import torch.nn.functional as F
import torch
from train import evaluate_epoch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from prettytable import PrettyTable


def predictChoice(model, dataset):
    """
    Predict final choice probability and label
    """
    with torch.no_grad():
        prob_choice = F.softmax(model.forward(dataset.z, dataset.x), dim=1)
    return prob_choice, prob_choice.argmax(dim=1)


def summarizeDataset(model, ds):
    dl = DataLoader(ds, batch_size=model.args.batch_size, shuffle=False, num_workers=5)

    # nll
    nll = evaluate_epoch(model, dl)

    # predict choice 
    prob_choice, pred_choice = predictChoice(model, ds)

    # choice accuracy
    acc_choice = accuracy_score(ds.y, pred_choice)

    sm = {"nll": nll, "acc": acc_choice, "acc_true": ds.acc, "nll_true": ds.nll}
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
    x.field_names = [""] + ["acc_" + data for data in data_list] + ["nll_" + data for data in data_list]
    x.add_row(["model"] + [np.round(summary[data]["acc"], precision) for data in data_list] + [
        np.round(summary[data]["nll"], precision) for data in data_list])
    x.add_row(["true"] + [np.round(summary[data]["acc_true"], precision) for data in data_list] + [
        np.round(summary[data]["nll_true"], precision) for data in data_list])
    return x


def predictTastes(model, z):
    with torch.no_grad():
        vots = model.taste(z)[:, 0]
        vowts = model.taste(z)[:, 1]
    return {'vots': vots, 'vowts': vowts}


def value_of_x(A0, A1, A2, Z):
    """
    Compute value of time for N persons given person characteristics z (N,D)
    Input:
        A0, A1, A2: 0, 1st and 2nd order interaction coefficients
        Z: person input (N,D)
    Return:
        vox: (N,1)
    """
    vox = A0 + torch.matmul(Z, A1) + torch.diag(torch.matmul(torch.matmul(Z, A2), Z.transpose(0, 1)))
    return vox


def RMSE_vector(pred, true):
    return ((((pred-true)**2).sum()/len(pred)) ** 0.5).item()


def ABSE_vector(pred, true):
    return (torch.abs(pred - true).sum()/len(pred)).item()


def RE_vector(pred, true):
    return ((torch.abs(pred-true)/torch.abs(true)).sum()/len(pred)).item()


def RMSE(pred_vots, true_vots):
    """
    pred_vots: list of pred_vots for train/dev/test
    """
    sum_error = 0.0
    count = 0
    for i in range(len(pred_vots)):
        sum_error += ((pred_vots[i] - true_vots[i]) ** 2).sum()
        count += len(pred_vots[i])
    rmse = (sum_error / count) ** 0.5
    return rmse


def ABSE(pred_vots, true_vots):
    sum_error = 0.0
    count = 0
    for i in range(len(pred_vots)):
        sum_error += (torch.abs(pred_vots[i] - true_vots[i])).sum()
        count += len(pred_vots[i])
    return sum_error / count


def RE(pred_vots, true_vots):
    """
    Relative error (pred-true)/true
    """
    sum_error = 0.0
    count = 0
    for i in range(len(pred_vots)):
        sum_error += (torch.abs(pred_vots[i] - true_vots[i]) / torch.abs(true_vots[i])).sum()
        count += len(pred_vots[i])
    return sum_error / count


def printError(rmse, mabse, re):
    s = ""
    s += "rmse:" + str(rmse) + "\n"
    s += "mean absolute error:" + str(mabse) + "\n"
    s += "percentage error:" + str(re) + "\n"
    return s


def plotVOT(pred_vots, true_vots, x_values, legend, result_path):
    fg, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    for vot in pred_vots:
        ax1.plot(x_values, vot.flatten().numpy())
    for vot in true_vots:
        ax2.plot(x_values, vot.flatten().numpy())
    ax1.set_title("Predicted")
    ax2.set_title("True")
    ax1.legend(legend)
    ax2.legend(legend)
    ax1.set_xlabel("income ($ per hour)", fontsize=12)
    ax1.set_ylabel("value of time ($ per hour)", fontsize=12)
    ax2.set_xlabel("income ($ per hour)", fontsize=12)
    ax2.set_ylabel("value of time ($ per hour)", fontsize=12)

    fg.savefig(result_path + "/" + "VOT_vs_inc.png", dpi=250)
    plt.close()


def plotVOWT(pred_vowts, true_vowts, x_values, legend, result_path):
    fg, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    for v in pred_vowts:
        ax1.plot(x_values, v.flatten().numpy())
    for v in true_vowts:
        ax2.plot(x_values, v.flatten().numpy())
    ax1.set_title("Predicted")
    ax2.set_title("True")
    ax1.legend(legend)
    ax2.legend(legend)
    ax1.set_xlabel("income ($ per hour)", fontsize=12)
    ax1.set_ylabel("value of waiting time ($ per hour)", fontsize=12)
    ax2.set_xlabel("income ($ per hour)", fontsize=12)
    ax2.set_ylabel("value of waiting time ($ per hour)", fontsize=12)

    fg.savefig(result_path + "/" + "VOWT_vs_inc.png", dpi=250)
    plt.close()


def plotLoss(loss_train_list, loss_dev_list, fig_path):
    plt.plot(loss_train_list)
    plt.plot(loss_dev_list)
    plt.legend(["loss_train", "loss_dev"])
    plt.xlabel("number of epochs")
    plt.ylabel("negative loglikelihood")
    plt.savefig(fig_path + "/loss_train_dev.png", dpi=300)

import torch


def RMSE(pred_vots, true_vots):
    '''
    pred_vots: dictionary {key:tensor value of time}
    '''
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
    '''
    Relative error (pred-true)/true
    '''
    sum_error = 0.0
    count = 0
    for i in range(len(pred_vots)):
        sum_error += (torch.abs(pred_vots[i] - true_vots[i]) / torch.abs(true_vots[i])).sum()
        count += len(pred_vots[i])
    return sum_error / count

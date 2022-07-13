import torch
import argparse
from torch.utils.data import DataLoader
from data_utils import ChoiceDataset
from models import TasteNetChoice, TasteNetChoiceSep
from train import train
from evaluation import summarize, printSummary, predictTastes, printError, RMSE, ABSE, RE, plotVOT, plotVOWT, plotLoss
from simulate import error_of_vot, error_of_vowt, dic_z, dic_z_z01, dic_z_zall, inc
import copy
import pickle
from regress_coef import regress

# Specify input parameters 
parser = argparse.ArgumentParser(description='TasteNet (Toy Example)')

# Training-related 
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, required=True, default=0.000)
parser.add_argument("--nll_tol", type=float, default=0.0001, help="tolerance for nll convergence")

## NN structure
parser.add_argument("--separate", action='store_true', help='enable separate networks to learn taste parameters')
parser.add_argument("--layer_sizes", nargs='+', required=True, default=[3, 1])
parser.add_argument("--activation", type=str, required=True, default=None)
parser.add_argument("--transform", type=str, default=None,
                    help="what kind of transform on the taste: exp, relu or empty string")

# ## For MNL specification only
parser.add_argument("--if_z01", type=bool, default=False, help="whether to add interaction z0*z1")
parser.add_argument("--if_zall", type=bool, default=False, help="whether to add all interactions among z")

parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')

# input data directory
parser.add_argument("--data_dir", type=str, default="../toy_data")
parser.add_argument("--data_file", type=str, default="data dictionary in pickle format, including train/dev/test")

parser.add_argument("--result_root", type=str, required=True)

# model run number
parser.add_argument("--model_no", type=int, required=True, default=0)

# =======Parse arguments=============
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

args.data_train = "train_" + args.N_train + ".pkl"
args.data_dev = "dev_" + args.N_train + ".pkl"
args.data_test = "test_" + args.N_train + ".pkl"


def str_of_arg(arg):
    if arg != "":
        return "_" + arg
    else:
        return ""


# input data name 
str_act = str_of_arg(args.activation)
str_transform = str_of_arg(args.transform)

str_H = ""
if len(args.layer_sizes) > 2:
    str_H = "_H" + "_".join(str(e) for e in args.layer_sizes[1:-1])

str_wd = ""
if args.weight_decay > 0:
    str_wd = "_" + str(args.weight_decay)

str_suffix = ""
if args.if_z01:
    str_suffix = "_z01"
if args.if_zall:
    str_suffix = "_zall"
args.scenario = "model" + str_H + str_act + str_transform + str_wd + str_suffix + "_no" + str(args.model_no)

args.output_dir = args.result_root + "/" + args.scenario

# ====== make output directory =====
import os

if not os.path.exists(args.result_root):
    os.mkdir(args.result_root)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if args.seed != None:
    torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

# load data
data = pickle.load(open(os.path.join(args.data_dir, args.data_file), "rb"))

ds_train = ChoiceDataset(data['train'])
ds_dev = ChoiceDataset(data['dev'])
ds_test = ChoiceDataset(data['test'])

data_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=5)
data_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=5)
data_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=5)

args.K, args.J = ds_train[0]["x"].size()
args.layer_sizes = [int(e) for e in args.layer_sizes]

# ======Get model==================
print(args)
if args.separate:
    model = TasteNetChoiceSep(args)
else:
    model = TasteNetChoice(args)

print(model)

model.args.init_params = model.getParams()
pickle.dump(model.args.init_params, open(args.output_dir + "/params_init.pkl", "wb"))

loss_train_list, loss_dev_list, best_model = train(model, data_train, data_dev, args)

# ======Plot loss=====================
plotLoss(loss_train_list, loss_dev_list, args.output_dir)

# ======Summary=====================
summary = summarize(best_model, ds_train, ds_dev, ds_test)
x = printSummary(summary, precision=3)
print(x)

print(model.getParams())

# save summary
f = open(args.output_dir + "/" + "summary_result.txt", "w")
f.write(str(x))
f.close()

# save model args
f = open(args.output_dir + "/" + "summary_args.txt", "w")
f.write(str(args))
f.close()

# save model string
f = open(args.output_dir + "/" + "summary_modelstr.txt", "w")
f.write(str(best_model))
f.close()

# ======= Model parameters ==========
params = best_model.getParams()
pickle.dump(params, open(args.output_dir + "/params.pkl", "wb"))

# ======Analysis Code=====================

result = {}

# ======Toy data VOT=====================
allds = {"train": ds_train, "dev": ds_dev, "test": ds_test}

pred_vots = [predictTastes(best_model, allds[name].z)['vots'] for name in ["train", "dev", "test"]]
true_vots = [allds[name].vots for name in ["train", "dev", "test"]]

rmse = RMSE(pred_vots, true_vots).item()
mabse = ABSE(pred_vots, true_vots).item()
re = RE(pred_vots, true_vots).item()

result['vot_rmse'] = rmse
result['vot_mabse'] = mabse
result['vot_re'] = re

s = ""
s += "rmse:" + str(rmse) + "\n"
s += "mean absolute error:" + str(mabse) + "\n"
s += "percentage error:" + str(re) + "\n"

f = open(args.output_dir + "/" + "vot_error.txt", "w")
f.write(str(s))
f.close()

pickle.dump(pred_vots, open(args.output_dir + "/" + "pred_vots.pkl", "wb"))

# =======Toy data VOWT =======
pred_vowts = [predictTastes(best_model, allds[name].z)['vowts'] for name in ["train", "dev", "test"]]
true_vowts = [allds[name].vowts for name in ["train", "dev", "test"]]

rmse = RMSE(pred_vowts, true_vowts).item()
mabse = ABSE(pred_vowts, true_vowts).item()
re = RE(pred_vowts, true_vowts).item()

result['vowt_rmse'] = rmse
result['vowt_mabse'] = mabse
result['vowt_re'] = re

s = ""
s += "rmse:" + str(rmse) + "\n"
s += "mean absolute error:" + str(mabse) + "\n"
s += "percentage error:" + str(re) + "\n"

f = open(args.output_dir + "/" + "vowt_error.txt", "w")
f.write(str(s))
f.close()

pickle.dump(pred_vowts, open(args.output_dir + "/" + "pred_vowts.pkl", "wb"))

# ======= VOT on simulated z ==========
if args.if_z01:
    input_z = copy.deepcopy(dic_z_z01)
elif args.if_zall:
    input_z = copy.deepcopy(dic_z_zall)
else:
    input_z = copy.deepcopy(dic_z)

# ======= Error of VOT on simulated z ==========
sim_pred_vots, sim_true_vots, rmse, mabse, re = error_of_vot(best_model, dic_z, input_z, ds_train.params)
f = open(args.output_dir + "/" + "sim_vot_error.txt", "w")
f.write(str(printError(rmse, mabse, re)))
f.close()

result['sim_vot_rmse'] = rmse
result['sim_vot_mabse'] = mabse
result['sim_vot_re'] = re

# ======= Plot VOT on simulated z ==========
plotVOT(sim_pred_vots, sim_true_vots, (inc * 60).numpy(), dic_z.keys(), args.output_dir)

# ======= Error of VOWT on simulated z ==========
sim_pred_vowts, sim_true_vowts, rmse, mabse, re = error_of_vowt(best_model, dic_z, input_z, ds_train.params)
f = open(args.output_dir + "/" + "sim_vowt_error.txt", "w")
f.write(str(printError(rmse, mabse, re)))
f.close()

result['sim_vowt_rmse'] = rmse
result['sim_vowt_mabse'] = mabse
result['sim_vowt_re'] = re

# ======= Plot VOWT on simulated z ==========
plotVOWT(sim_pred_vowts, sim_true_vowts, (inc * 60).numpy(), dic_z.keys(), args.output_dir)

# ====== Regression ======
coefs_asc1 = model.getParams()[-1][0].item()
coefs_time, coefs_wait = regress(best_model, dic_z, dic_z_zall)
coefs_pred = [coefs_asc1] + coefs_time + coefs_wait

coefs_asc1_true = ds_train.asc1
coefs_time_true = ds_train.coefs_time
coefs_wait_true = ds_train.coefs_wait
coefs_true = [coefs_asc1_true] + coefs_time_true + coefs_wait_true


import numpy as np
print("True coef")
print(np.round(np.array(coefs_true), 4))

print("Predicted coef")
print(np.round(np.array(coefs_pred), 4))

from evaluation import RMSE_vector, ABSE_vector, RE_vector
coefs_pred = torch.Tensor(coefs_pred)
coefs_true = torch.Tensor(coefs_true)

rmse = RMSE_vector(coefs_pred, coefs_true)
mabse = ABSE_vector(coefs_pred, coefs_true)
re = RE_vector(coefs_pred, coefs_true)

print(rmse, mabse, re)
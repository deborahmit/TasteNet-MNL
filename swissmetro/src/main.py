import argparse
import torch
from torch.utils.data import DataLoader
from data_utils import ChoiceDataset
from models import ChoiceFlex
from train import train
from evaluation import summarize, printSummary, predictParams, bToDataFrame
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict


# Specify input parameters 
parser = argparse.ArgumentParser(description='Neural network for Flexible utility (VOT =f(z))')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--num_epochs', type=int, default=100, 
                    help='number of epochs to train (default: 100)')
parser.add_argument("--nll_tol", type=float, default=0.001, help="tolerance for nll convergence")
parser.add_argument("--no_chg", type=int, default=5, help="no improvement on dev loss")

parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, required=False, default=0.000)
parser.add_argument("--transform", type=str, default="exp") # other options: "none", "relu"
#parser.add_argument("--penalty_constraint", type=float, default=0.00)

parser.add_argument("--l1", type=float, required=True, default=0.000) # l1 penalty only on non-bias coef 
parser.add_argument("--l2", type=float, required=True, default=0.000) # l2 penalty only on non-bias coef 

## network structure
parser.add_argument("--taste_params", type=int, default=8) # number of taste parameters, output size of the last hidden layer , fixed
parser.add_argument("--K", type=int, default=4) # input size of choice model (alternative attributes, including intercept 1) 
parser.add_argument("--J", type=int, default=3) # number of alternatives

parser.add_argument("--hidden_sizes", nargs='+', default=[]) #hidden layer sizes for the taste module 
parser.add_argument("--act_func", type=str, default="")
parser.add_argument("--mu", type=float, default=1.0) #shape of the exp transform function 

# model no
parser.add_argument("--model_no", type=int, default=999, required=False, help="model no")

parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=None, 
                    help='random seed (default: None)')

# directory
parser.add_argument("--data_path", type=str, default="../data", help="directory name for input data")
parser.add_argument("--result_root", type=str, default='../results', help="root directory for model results")

#=======Parse arguments=============
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# input data name 
args.data_train = "train.pkl"
args.data_dev = "dev.pkl"
args.data_test = "test.pkl"
args.data_all = "swissmetro_all.pkl"


#======Prepare input data ========
ds_train = ChoiceDataset(args.data_path, args.data_train)
ds_dev = ChoiceDataset(args.data_path, args.data_dev)
ds_test = ChoiceDataset(args.data_path, args.data_test)
ds_all = ChoiceDataset(args.data_path, args.data_all)

args.z_size = ds_train.z.size()[1]

data_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=5)
data_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=5)
data_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=5)


#======Prepare output path ========

# Related to model scenario folder 
args.hidden_sizes = [int(e) for e in args.hidden_sizes]
args.layer_sizes = [args.z_size]+args.hidden_sizes + [args.taste_params]

str_hid_layers = ""
str_hidden_sizes = ""
if len(args.hidden_sizes) > 0:
    str_hid_layers = "_" + str(len(args.hidden_sizes)) + "_hid_layers"
    str_hidden_sizes = "_H" + "_".join(str(e) for e in args.hidden_sizes)

str_l1 = ""
if args.l1 > 0: 
    str_l1 = "_l1_" + str(args.l1)

str_l2 = ""
if args.l2 > 0: 
    str_l2 = "_l2_" + str(args.l2)

str_act = ""
if args.act_func!="":
    str_act = "_"+args.act_func

str_transform = ""
if args.transform != "exp": # default is exp transform, if not default, put it in the name of the scenario 
    str_transform = "_" + args.transform

# Model result path
args.result_path = args.result_root +"/model"+ str_hidden_sizes + str_act + str_transform + str_l1 + str_l2  +  "_no" + str(args.model_no)

#====== make output directory =====
import os
if not os.path.exists(args.result_root):
    os.mkdir(args.result_root)
if not os.path.exists(args.result_path):
    os.mkdir(args.result_path)


if args.seed != None: 
    torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

#======Get model==================
print(args)
model = ChoiceFlex(args)
print(model)

loss_train_list, loss_dev_list, best_model = train(model, data_train, data_dev, args)

summary = summarize(best_model, ds_train, ds_dev, ds_test)
pickle.dump(summary, open(args.result_path + "/" + "summary.pkl", "wb"))

x = printSummary(summary, precision=3)
print(x)

# save summary
f = open(args.result_path + "/" + "summary_result.txt", "w")
f.write(str(x))
f.close()

# save model args 
f = open(args.result_path + "/" + "summary_args.txt", "w")
f.write(str(args))  
f.close()

# save model string
f = open(args.result_path + "/" + "summary_modelstr.txt", "w")
f.write(str(model))  
f.close()

# save predicted taste parameters (b) 
b = predictParams(best_model, ds_all.z)
pickle.dump(b, open(args.result_path + "/" + "b.pkl", "wb"))

b_names = ['TRAIN_TT', 'SM_TT', 'CAR_TT', 'TRAIN_HE', 'SM_HE', 'SM_SEATS', 'TRAIN_ONE', 'SM_ONE']
b_mean = OrderedDict(zip(b_names,b.mean(0)))
b_std = OrderedDict(zip(b_names,b.std(0)))

b_max = OrderedDict(zip(b_names,b.max(0)[0]))
b_min = OrderedDict(zip(b_names,b.min(0)[0]))

b_mean = bToDataFrame(b_mean)
b_std = bToDataFrame(b_std)
b_max = bToDataFrame(b_max)
b_min = bToDataFrame(b_min)

b_mean.to_csv(args.result_path + "/" + "b_mean.csv")
b_std.to_csv(args.result_path + "/" + "b_std.csv")
b_max.to_csv(args.result_path + "/" + "b_max.csv")
b_min.to_csv(args.result_path + "/" + "b_min.csv")

print ("\nb_mean")
print (b_mean)

print ("\nb_std")
print (b_std)

print ("\nb_max")
print (b_max)

print ("\nb_min")
print (b_min)

## print and save parameter mean values
f = open(args.result_path + "/" + "summary_params_mean_std.txt", "w")
f.write("all data mean\n" + str(b_mean) + "\n") 
f.write("all data std\n" + str(b_std) + "\n") 
f.close()

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import copy 
import pickle


def train(model, data_train, data_dev, args):
    """
    Run num_epochs training epochs, and evaluate on data_dev at the end of each epoch
    """
    optimizer =  optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    best_model = None
    best_dev_loss = 10000 # BIG number 
    count_no_chg = 0  # track dev_loss if there is change  
    loss_train_list, loss_dev_list = [], []

    for epoch in range(1,args.num_epochs+1):
        train_epoch(model,data_train, optimizer)
        loss_train = evaluate_epoch(model, data_train)
        loss_dev = evaluate_epoch(model, data_dev)
        loss_train_list.append(loss_train)
        loss_dev_list.append(loss_dev)
        
        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, loss_train))
        print('====> Epoch: {} Dev loss: {:.4f}'.format(epoch, loss_dev))
        
        if loss_dev < best_dev_loss:
            best_dev_loss = loss_dev
            best_model = copy.deepcopy(model)
            count_no_chg = 0
        else:
            count_no_chg += 1
        
        if count_no_chg >=args.no_chg and (np.abs(average_diff(loss_dev_list[-args.no_chg:]))<=args.nll_tol):
            break
        
    pickle.dump(best_model, open(args.result_path + "/best_model.pkl", "wb"))
    pd.DataFrame(np.array([loss_train_list,loss_dev_list]).T,\
                 columns = ["train_loss", "dev_loss"]).to_csv(args.result_path + "/train_dev_loss.csv", index=True)
    
    return loss_train_list, loss_dev_list, best_model


def average_diff(list_of_values):
    return (np.array(list_of_values[:-1])-np.array(list_of_values[1:])).mean()


def train_epoch(model, data_train, optimizer):
    '''
    Run 1 forward pass through the data 
    Parameters:
        data_train: training data DataLoader
        optimizer: 
    '''
    model.train()
    sum_loss_all = 0.0
    sum_loss = 0.0
    batches= 0
    
    # iterate over batches
    for batch_idx, data in enumerate(data_train):
        # forward pass
        batch_size = len(data["y"])
        prob, loss_constraint = model.forward(data["z"],data["x"], data["av"])
        
        likelihood = prob.gather(1,data["y"].reshape(batch_size, 1))
        loss = -torch.log(likelihood).sum()/batch_size
        sum_loss += loss.item()
        
        # if l1 or l2 penalty, add penalty to loss
        if model.args.l2 > 0:
            loss = loss + model.L2Norm() * model.args.l2
        if model.args.l1 > 0: 
            loss = loss + model.L1Norm() * model.args.l1

        # back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss, batch size 
        sum_loss_all += loss.item()
        batches += 1 
    return sum_loss_all / batches, sum_loss / batches


def evaluate_epoch(model, data_loader):
    '''
    Loss over data 
    
    Parameters:
        model: 
        data_loader: data to evaluate loss (DataLoader)
    '''
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            batch_size = len(data["y"])
            prob, loss_constraint = model.forward(data["z"], data["x"], data["av"])
            likelihood = prob.gather(1,data["y"].reshape(batch_size, 1))
            loss = -torch.log(likelihood).sum()/batch_size
            total_loss += loss.item()
            batches += 1
    return total_loss / batches
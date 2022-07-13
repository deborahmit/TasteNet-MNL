import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import copy 
import os


def train(model, data_train, data_dev, args, save=False):
    """
    Run num_epochs training epochs, and evaluate on data_dev at the end of each epoch
    """
   
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model = None
    best_dev_loss = 10000  # BIG number
    count_no_chg = 0  # track dev_loss if there is change

    loss_train_list, loss_dev_list = [],[]
    
    for epoch in range(1, args.num_epochs+1):
        train_epoch(model, data_train, optimizer)
        
        loss_train = evaluate_epoch(model, data_train)
        loss_dev = evaluate_epoch(model, data_dev)
        loss_train_list.append(loss_train)
        loss_dev_list.append(loss_dev)
        
        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, loss_train))
        print('====> Epoch: {} Dev loss: {:.4f}'.format(epoch, loss_dev))

        if epoch > 10 and loss_train > 1:  # early termination if this run far from the solution
            return None, None, None, False

        if loss_dev < best_dev_loss:
            best_dev_loss = loss_dev
            best_model = copy.deepcopy(model)
            count_no_chg = 0
        else:
            count_no_chg += 1
        
        if count_no_chg >= 10:
            break

    if save:
        pickle.dump(best_model, open(args.output_dir + "/best_model.pkl", "wb"))
        pd.DataFrame(np.array([loss_train_list, loss_dev_list]).T,
                     columns=["train_loss", "dev_loss"]).to_csv(os.path.join(args.output_dir + "train_dev_loss.csv"), index=True)

    return loss_train_list, loss_dev_list, best_model, True


def train_epoch(model, data_train, optimizer):
    """
    Run 1 forward pass through the data
    Parameters:
        data_train: training data DataLoader
        optimizer:
    """
    model.train()
    total_loss = 0.0
    batches= 0

    # iterate over batches
    for batch_idx, data in enumerate(data_train):
        # forward pass
        loss = F.cross_entropy(model.forward(data["z"], data["x"]), data["y"]) # input of cross_entropy is before taking log_softmax! batch average loss
        # back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss, batch size 
        total_loss += loss.item()
        batches += 1 
    return total_loss / batches 


def evaluate_epoch(model, data_loader):
    """
    Loss over data

    Parameters:
        model:
        data_loader: data to evaluate loss (DataLoader)
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            loss = F.cross_entropy(model.forward(data["z"], data["x"]), data["y"])
            total_loss += loss.item()
            batches += 1
    return total_loss / batches
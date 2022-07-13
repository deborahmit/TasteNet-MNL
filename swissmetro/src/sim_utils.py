import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

mode_loc = {"train": 0, "sm": 1, "car":2}  # col number of each mode 
taste_loc = {"vott": 0, "vohw": 1, "seats":2,"cost": 3, "asc": 4} # row number of each taste_name
z_dic = {"MALE": {0:"Female", 1:"Male"}, \
 "AGE": {0: "(0,24]", 1:"(24,39]", 2: "(39,54]", 3: "(54,65]", 4: "(65,)"},\
 "INCOME": {0:"under 50", 1: "50 to 100", 2: "over 100", 3: "unknown"},\
 "FIRST": {0: "Not first-class", 1: "First-class"},\
 "WHO": {0: "self", 1: "employer", 2: "half-half"},\
 "PURPOSE": {0:"Commute", 1:"Shopping", 2:"Business", 3:"Leisure"},\
 "LUGGAGE": {0:"None", 1: "One piece", 2: "Several pieces"},\
 "GA": {0: "GA: No", 1:"GA: Yes"}
}

taste_fullname = {"vott": "Value of Time", "vohw":"Value of Headway", "seats": "SM: Airline seats", "cost": "Cost", "asc": "Alternative Specific Constant"}


def toString(l1,l2):
    if l1>0:
        return "_l1_" + str(l1)
    if l2>0:
        return "_l2_" + str(l2)
    else:
        return ""


def getTaste(mode, taste_name, z_name, b, z_all, z_all_names, mode_loc, taste_loc):
    '''
    Select taste parameter from estimated b based on search query: mode, taste_name, z_name
        # Data input 
        b: taste parameters for all obs in Swissmetro data (N,K,J)
        z: N obs's input z
        z_names: list of z_name corresponding to dimensions of z
        
        # Selection criteria: 
        mode: "train", "sm", "car"
        taste_name: "vott", "vohw", "asc" 
        z_name: pop segment to select, such as "MALE_0"
    ''' 
    ind = z_all_names.index(z_name)
    selected = z_all[:,ind]==1
    b_sel = b[selected,taste_loc[taste_name], mode_loc[mode]]
    if taste_name != "asc" and taste_name != "seats": # vot: return absolute values
        return -torch.Tensor(b_sel) # return the absolute value 
    return torch.Tensor(b_sel)


def cdf(data):
    '''
    plot cdf for data
    '''
    # Set bins edges: each obs falling into one bin
    data_sorted=sorted(data)
    x, counts = np.unique(data_sorted, return_counts=True)
    y = np.cumsum(counts).astype(float)/np.sum(counts)
    
    # Plot the cdf
    sns.lineplot(x, y)
    plt.ylim((0,1))
    plt.ylabel("CDF")
    plt.grid(True)
    

def plotTasteByZ_CDF(mode, taste_name, z_group, b, z, z_names, z_levels, mode_loc, taste_loc, save, fig_dir):
    fg = plt.figure()
    tastes = {}
    
    # Get tastes by group level
    for level in range(z_levels[z_group]):
        z_name = z_group+"_"+str(level)  # e.g. "MALE_0", "MALE_1"
        tastes[z_name] = getTaste(mode, taste_name, z_name, b, z, z_names, mode_loc, taste_loc)
        # tastes[z_name] = tastes[z_name].view(-1,9)[:,0] # 9 menus for the same person
    # Plot cdf for all levels for z
    for level in range(z_levels[z_group]):
        z_name = z_group + "_" + str(level)
        cdf(tastes[z_name].numpy())
    
    # Add labels
    plt.title(taste_fullname[taste_name] +" (" + mode + ")")
    plt.legend([str(i) + " : " + z_dic[z_group][i] for i in z_dic[z_group]])
    if taste_name != "asc" and taste_name != "seats":
        plt.xlabel(taste_fullname[taste_name] + " (CHF/min)")
    else: 
        plt.xlabel(taste_fullname[taste_name])
    plt.ylabel("Cumulative Probability Density")
    
    # save figure 
    if save:
        plt.savefig(fig_dir + "/" + mode + "_" + taste_name + "_by_" + z_group + "_CDF", dpi=300)
    return tastes # tastes by z_group 

def pdf(data):
    '''
    Plot PDF
    '''
    data_size=len(data)
    bins = np.arange(min(data),max(data),0.1)
    sns.distplot(data, bins, kde=False, norm_hist=True)
    ## note: # i.e., the area (or integral) under the histogram will sum to 1. This is achieved by dividing the
    ## count by the number of observations times the bin width and not dividing by the total number of observations. 
    plt.ylabel("PDF")
    plt.grid(True)


def plotTasteByZ_PDF(mode, taste_name, z_group, b, z, z_names, z_levels, mode_loc, taste_loc, save, fig_dir):
    fg = plt.figure()
    tastes = {}
    # Get tastes by group level
    for level in range(z_levels[z_group]):
        z_name = z_group+"_"+str(level)  # e.g. "MALE_0", "MALE_1"
        tastes[z_name] = getTaste(mode, taste_name, z_name, b, z, z_names, mode_loc, taste_loc)

    # Plot pdf for all levels for z
    for level in range(z_levels[z_group]):
        z_name = z_group + "_" + str(level)
        pdf(tastes[z_name].numpy())
    
    # Add labels
    plt.title(taste_fullname[taste_name] + " (" + mode + ")")
    plt.legend([str(i) + " : " + z_dic[z_group][i] for i in z_dic[z_group]])
    if taste_name != "asc" and taste_name != "seats":
        plt.xlabel(taste_fullname[taste_name] + " (CHF/min)")
    else: 
        plt.xlabel(taste_fullname[taste_name])
    plt.ylabel("Probability Density")
    
    # save figure 
    if save:
        plt.savefig(fig_dir + "/" + mode + "_" + taste_name + "_by_" + z_group + "_PDF", dpi=300)
    return tastes # tastes by z_group 


def plotTasteByMode_CDF(taste_name, b, mode_loc, taste_loc, save, fig_dir):
    tastes = {}
    labels = []
    for mode in mode_loc:
        tastes[mode] = b[:,taste_loc[taste_name], mode_loc[mode]]
        if not (mode=="car" and (taste_name=="vohw" or taste_name=="asc")):
            cdf(-tastes[mode].numpy()) # vot absolute value
            labels.append(mode)
    
    # Add labels
    plt.title(taste_fullname[taste_name] +" by Mode")
    plt.legend(labels)
    
    if taste_name != "asc":
        plt.xlabel(taste_fullname[taste_name] + " (CHF/min)")
    else: 
        plt.xlabel(taste_fullname[taste_name])
    plt.ylabel("Cumulative Probability Density")

    # save figure 
    if save:
        plt.savefig(fig_dir + "/" + taste_name + "_by_mode_CDF", dpi=300)
    return tastes # tastes by z_group 

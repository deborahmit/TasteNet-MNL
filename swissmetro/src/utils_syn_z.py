import torch
import pandas as pd


def sampleZ(z_levels, N):
    '''
    Randomly generate N samples for input z
    Returns 
        z: (N,7) where 7 is the number of z(categorical)
    '''
    num_z = len(z_levels) # z dimension
    z = torch.zeros(N,num_z)
    i = 0
    for key in z_levels:
        z[:,i] = torch.randint(0, z_levels[key], size=(N,))
        i+=1
    return z


def cat2dummies(df_z, var, num_levels):
    '''
    Convert z to dummies 
    '''
    dmy = pd.get_dummies(df_z[var])
    dmy.columns = [var+str("_")+str(i) for i in range(num_levels)]
    df_z = pd.concat([df_z, dmy], axis=1)
    return df_z


def generateZInput(N_samples, z_levels):
    D = len(z_levels)
    z_samples = sampleZ(z_levels, N_samples)
    df_z = pd.DataFrame(z_samples.numpy(), columns = z_levels.keys())
    
    for var in z_levels:
        df_z = cat2dummies(df_z, var, z_levels[var])
    
    z_input = torch.Tensor(df_z[df_z.columns[D:]].values)
    return z_input


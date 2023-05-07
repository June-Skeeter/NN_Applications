# Miscellaneous functions to keep the presentation code cleaner 
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

def Calc_VPD(X,Y = None):
    # Calculate vapour pressure (hPa)
    # From TA (in celsius) and RH %
    # Can accept tensorflow tensor or numpy arrys, or pandas sereis
    # Generally woudn't use tensors for something like this, but setup as an example
    # Using tensors allows us to caclate the partial derivatives of the function
    if not tf.is_tensor(X):
        if Y is None:
            TA,RH = X[:,0],X[:,1]
        else:
            TA,RH = X,Y
        ea_H = 0.61365*np.exp((17.502*TA)/(240.97+TA))
        e_H = RH*ea_H/100
        VPD = (ea_H - e_H)*10
    else:
        ea_H = 0.61365*tf.exp((17.502*X[:,0])/(240.97+X[:,0]))
        e_H = X[:,1]*ea_H/100
        VPD = (ea_H - e_H)*10
    return (VPD)


def Create_Grid(x1,x2,func=None):  
    # Create a 2d Grid of 2 X variables
    # Option to create grid of Y as function of X
    grid_x1,grid_x2 = np.meshgrid(x1,x2)
    flat_X=np.array([grid_x1.flatten(),grid_x2.flatten()])
    if func != None:
        flat_Y = func(flat_X[0],flat_X[1])
        grid_y = flat_Y.reshape(grid_x1.shape)
    else:
        grid_y = None
    return(flat_X,grid_x1,grid_x2,grid_y)


def byInterval(df,x,Vars,bins=None,agg='mean'):
    # Aggregates data by an interval and returns a 95% CI
    # If dataframe has a datetime index, use resample
    # Handles all other data types by groubpby
    # Will group by continuous data by intervals if fed bins
    # Otherwise groupby will treat index as discrete data points
    STD = [var+'_std' for var in Vars]
    c = [var+'_c' for var in Vars]
    CI = [var+'_CI95' for var in Vars]
    if isinstance(df.index, pd.DatetimeIndex):
        Set = df.resample(x).agg(agg)
        Set[STD] = df.resample(x).std(numeric_only=True)[Vars]
        Set[c] = df.resample(x).count()[Vars]
    else:
        if bins is None:
            df[f"{x}_grp"] = df[x].copy()
        else:
            df['bins'] = pd.cut(df[x],bins=bins)
            df[f"{x}_grp"] = df['bins'].apply(lambda x: x.mid)
            df = df.drop('bins',axis=1)        
        Set = df.groupby(f'{x}_grp').agg(agg,numeric_only=True)
        Set[STD] = df.groupby(f'{x}_grp').std(numeric_only=True)[Vars]
        Set[c] = df.groupby(f'{x}_grp').count()[Vars]
    Set[CI] = Set[STD].values/(Set[c].values**.5)*(stats.t.ppf(0.95,Set[c].values))
    Set = Set.loc[Set[c].sum(axis=1)>len(c)*10]
    return Set[Vars+CI+STD+c]

def makeSplit(df,Mask=None,dropOut=.33,return_Full=False):
    # Add Random or Systematic gaps to the dataset
    # If no mask - drop randomly using dropOut rate
    # If Mask, drop values within bound(s) of mask(s)
    if Mask is None:
        Training = df.sample(frac=(1-dropOut))
        Validation = df.loc[df.index.isin(Training.index)==False].copy()
    else:
        Validation = pd.DataFrame()
        for key,mask in Mask.items():
            for m in mask:
                Validation = pd.concat([Validation,df.loc[df[key].between(m[0],m[1])]])
            Training = df.loc[df.index.isin(Validation.index)==False].copy()
    if return_Full == False:
        return(Training,Validation)
    else:
        return(Training,df)

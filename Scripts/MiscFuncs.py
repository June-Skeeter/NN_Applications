# Miscellaneous functions to keep the presentation code cleaner 

import numpy as np
import pandas as pd
import tensorflow as tf

def Calc_VPD(X,Y = None):
    # Calculate vapour pressure (hPa)
    # From TA (in celsius) and RH %
    # Can accept tensorflow tensor or numpy arrys, or pandas sereis
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


def byInterval(df,x,y,bins=None,agg='mean'):
    # Aggregates data by an interval and returns a 95% CI
    # If dataframe has a datetime index, use resample
    # Handles all other data types by groubpby
    # Will group by continuous data by intervals if fed bins
    # Otherwise groupby will treat index as discrete data points
    if isinstance(df.index, pd.DatetimeIndex):
        Set = df.resample(x).agg(agg)
        Set['std'] = df.resample(x).std()[y]
        Set['c'] = df.resample(x).count()[y]
    else:
        if bins is None:
            df[f"{x}_grp"] = df[x].copy()
        else:
            df['bins'] = pd.cut(df[x],bins=bins)
            df[f"{x}_grp"] = df['bins'].apply(lambda x: x.mid.round())
            df = df.drop('bins',axis=1)        
        Set = df.groupby(f'{x}_grp').agg(agg,numeric_only=True)
        Set['std'] = df.groupby(f'{x}_grp').std()[y]
        Set['c'] = df.groupby(f'{x}_grp').count()[y]
    Set['CI95'] = Set['std']/(Set['c']**.5)*1.96
    return(Set[[y,'std','c','CI95']],x,y)

def makeGap(df,Y=None,Mask=None,dropOut=.33):
    if Mask is None:
        Masked = df.sample(frac=dropOut)
        Dropped = df.loc[df.index.isin(Masked.index)==False].copy()
    elif Mask.ndim==1:
        Masked = df.loc[~df[Y].between(Mask[0],Mask[1])]
        Dropped = df.loc[df.index.isin(Masked.index)==False].copy()
    elif Mask.ndim==2:
        Dropped = pd.DataFrame()
        for mask in Mask:
           Dropped = pd.concat([Dropped,df.loc[df[Y].between(mask[0],mask[1])]])
        Masked = df.loc[df.index.isin(Dropped.index)==False].copy()
    
    return(Masked,Dropped)

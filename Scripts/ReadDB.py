import os
import numpy as np
import pandas as pd

def get_Traces(Site,Traces,Dir = '/mnt/w/'):
    ## Read a set of traces from the micromet database and add them to a dataframe
    ## Assumees the micromet Database folder is mapped to /mnt/w/
    ## Defaults to searching Clean/SecondStage/, can be altered as needed

    Data = pd.DataFrame()
    Time_Trace = 'Clean/SecondStage/clean_tv'

    for year in range (2017,2023):
        if os.path.exists(f'{Dir}{str(year)}/{Site}/'):
            filename = f'{Dir}{str(year)}/{Site}/{Time_Trace}'
            with open(filename, mode='rb') as file:
                Time = np.fromfile(file, 'float64')
                Time = pd.to_datetime(Time-719529,unit='D').round('T')
                D_trace = {}
                for Trace in Traces:
                    filename = f'{Dir}{str(year)}/{Site}/{Trace}'
                    with open(filename, mode='rb') as file:
                        trace = np.fromfile(file, 'float32')
                        D_trace[Trace]=trace
                df = pd.DataFrame(index=Time,data=D_trace)
                Data = pd.concat([Data,df])
    Data = Data.dropna(axis=0, how='all')
    Data.index.name='TimeStamp'
    return(Data)

# def filterFlux(df,F)

class filterFlux():

    def __init__(self,df,F):

        self.df = df.copy()
        self.F = F

    def dir_mask(self,dir,mask):
        for m in mask:
            self.df.loc[self.df[dir].between(m[0],m[1]),self.F]=np.nan
    
    def QA_QC(self,thresh=0):
        for f in self.F:
            self.df.loc[self.df[f'qc_{f}']>thresh,f]=np.nan

    def rain(self,P,thresh=0):
        self.df.loc[self.df[P]>thresh,self.F]=np.nan

    def MAD(self,z=5):
        for f in self.F:
            di = self.df[f].diff()-self.df[f].diff(-1)
            md = di.median()
            MAD = ((di-md).abs()).median()
            range = [md-((z*MAD)/0.675),md+((z*MAD)/0.675)]
            self.df.loc[~self.df[f].between(range[0],range[1]),f]=np.nan

    def uStar(self,u_star,u_thresh=.1):
        for f in self.F:
            self.df.loc[self.df[u_star]<u_thresh]=np.nan


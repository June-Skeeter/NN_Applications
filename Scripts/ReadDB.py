import os
import numpy as np
import pandas as pd

def get_Traces(Site,Traces,Dir = '/mnt/w/',sub_dir='Clean/SecondStage/'):
    ## Read a set of traces from the micromet database and add them to a dataframe
    ## Assumees the micromet Database folder is mapped to /mnt/w/
    ## Defaults to searching Clean/SecondStage/, can be altered as needed

    Data = pd.DataFrame()
    Time_Trace = 'clean_tv'

    for year in range (2015,2024):
        print(f'{Dir}{str(year)}/{Site}/')
        if os.path.exists(f'{Dir}{str(year)}/{Site}/'):
            filename = f'{Dir}{str(year)}/{Site}/{sub_dir}{Time_Trace}'
            print(filename)
            with open(filename, mode='rb') as file:
                Time = np.fromfile(file, 'float64')
                Time = pd.to_datetime(Time-719529,unit='D').round('T')
                D_trace = {}
                for Trace in Traces:
                    filename = f'{Dir}{str(year)}/{Site}/{sub_dir}{Trace}'
                    with open(filename, mode='rb') as file:
                        trace = np.fromfile(file, 'float32')
                        D_trace[Trace]=trace
                df = pd.DataFrame(index=Time,data=D_trace)
                Data = pd.concat([Data,df])
    Data = Data.dropna(axis=0, how='all')
    Data.index.name='TimeStamp'
    return(Data)


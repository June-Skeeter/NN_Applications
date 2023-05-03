# Train and run a simple random forest model
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


def RF_mod(Data,ax=None,missing=None,unit=None):
    # Set 10% of training data aside for validation
    # The RF model bootstraps the training data (train/test splitting)
    # SO the validation data will give an "independent" assessment
    T1 = time.time()
    RF = RandomForestRegressor()
    RF.fit(Data['X_train'],Data['Y_train'])
    T2 = time.time()
    print('Training Time:\n', np.round(T2 - T1,2),' Seconds')

    y_pred = RF.predict(Data['X_eval'])
    R2 = metrics.r2_score(Data['Y_eval'],y_pred)
    RMSE = metrics.mean_squared_error(Data['Y_eval'],y_pred)
    print()
    print('Validation metrics: \nr2 = ',
            np.round(R2,5),'\nRMSE = ',np.round(RMSE,5))
    print()

    
    Output = pd.DataFrame(data = {
            'target':Data['Y_eval'],
            'y_bar':y_pred,
        })

    return (Output,RF)

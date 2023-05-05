## Created by Dr. June Skeeter

## To DO
#  - save config & training data?
#  - expand to handle multiple targets?

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
from keras.models import model_from_json
from sklearn.ensemble import RandomForestRegressor

def clean_dir(dirname):
    if os.path.isdir(dirname):
        for (_, _, filenames) in os.walk(dirname):
            for f in filenames:
                os.remove(dirname+'/'+f)
    else:
        os.mkdir(dirname)
          
def make_Dense_model(config):
    clean_dir(config['Name'])
    input_layer = keras.layers.Input(len(config['inputs']))

    if config['Norm'] == True:
        norm = keras.layers.Normalization(mean=config['mean'],variance=config['variance'])
        x = norm(input_layer)
    else:
        x = input_layer

    for node,activation in zip(config['Nodes'],config['Activation']):
        x = keras.layers.Dense(node,
                               activation=activation,
                               kernel_initializer="glorot_uniform",
                               bias_initializer="zeros"
                               )(x)
    output_layer = keras.layers.Dense(len(config['target']))(x)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer,name=config['Name'])
    print(model.summary())
    model_json = model.to_json()
    with open(f"{config['Name']}/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

def train_model(config,Data): 
    T1 = time.time()
    with open(f"{config['Name']}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)
        model_architecture = model_from_json(json.dumps(architecture))

    for i in range(config['N_models']):
        model = keras.models.clone_model(model_architecture)
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"{config['Name']}/model_weights"+str(i)+".h5",save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=0),
        ]

        model.compile(optimizer="adam",loss="mean_squared_error",)

        model.fit(Data['X'],Data['Y'],callbacks=callbacks,validation_split=0.2,verbose=0,batch_size=config['batch_size'],epochs=config['epochs'])

        tf.keras.backend.clear_session()
        
    T2 = time.time()
    print('Training Time:\n', np.round(T2 - T1,2),' Seconds')
    
    if config['RF_comp']==True:
        Train_RF(config,Data)

def get_SSD(dy_dx):
    N =  dy_dx.shape[0]
    SSD = ((dy_dx)**2).sum(axis=1).mean(axis=0)
    if N > 1:
        SSD_CI = ((dy_dx)**2).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
    else:
        SSD_CI = SSD*np.nan
    return(SSD,SSD_CI)

def run_Model(config,Data):
    T1 = time.time()
    with open(f"{config['Name']}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)

    full_out = {'y_pred':[],'dy_dx':[]}
    
    if config['Norm'] == True and 'Normalization' in json.dumps(architecture):
        full_out['dy_dx_norm'] = []
    else:
        config['Norm'] = False

    ## Find all .h5 files (model weights)
    N = 0
    for (_, _, filenames) in os.walk(config['Name']):
        for f in filenames:
            if f.split('.')[-1]=='h5':
                N += 1
                loaded_model = model_from_json(json.dumps(architecture))
                loaded_model.load_weights(f"{config['Name']}/{f}")
                loaded_model.compile(loss='mean_squared_error', optimizer='adam')
                if not tf.is_tensor(Data['X']):
                    x = tf.convert_to_tensor(Data['X'])
                else:
                    x = Data['X']
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x)
                    est = loaded_model(x)
                full_out['y_pred'].append(est.numpy())
                full_out['dy_dx'].append(tape.gradient(est,x).numpy())
                
                # Scale derivatives by the variance from the Normalization layer
                # So they can be compared/ranked on the same scale
                if config['Norm'] == True:
                    for mod in architecture['config']['layers']:
                        if mod['class_name'] == 'Normalization':
                            v = np.array(mod['config']['variance'])
                            full_out['dy_dx_norm'].append((full_out['dy_dx'][-1]*v**.5))
                            break
    if N > 1:
        t_score = stats.t.ppf(0.95,N)
    else:
        t_score = np.nan

    for key in full_out.keys():
        full_out[key]=np.array(full_out[key])

    Mean_Output = pd.DataFrame(data = {
            'target':Data['Y'],
            'y_bar':full_out['y_pred'].mean(axis=0).flatten(),
            'y_CI95':full_out['y_pred'].std(axis=0).flatten()/(N)**.5*t_score
        })

    for i,xi in enumerate(config['inputs']):
        Mean_Output[f'{xi}']=Data['X'][:,i]
        Mean_Output[f'dy_d{xi}']=full_out['dy_dx'].mean(axis=0)[:,i]
        Mean_Output[f'dy_d{xi}_CI95']=full_out['dy_dx'].std(axis=0)[:,i]/(N)**.5*t_score

    SSD,SSD_CI = get_SSD(full_out['dy_dx'])
    RI = pd.DataFrame(index=config['inputs'],data = {'RI_bar':SSD/SSD.sum(),'RI_CI95':SSD_CI/SSD.sum()})
        
    if "dy_dx_norm" in full_out:
        for i,xi in enumerate(config['inputs']):
            Mean_Output[f'dy_d{xi}_norm']=full_out['dy_dx_norm'].mean(axis=0)[:,i]
            Mean_Output[f'dy_d{xi}_norm_CI95']=full_out['dy_dx_norm'].std(axis=0)[:,i]/(N)**.5*t_score
        SSD,SSD_CI = get_SSD(full_out['dy_dx_norm'])
        RI['RI_norm_bar']=SSD/SSD.sum()
        RI['RI_norm_CI95']=SSD_CI/SSD.sum()

    R2 = metrics.r2_score(Mean_Output['target'],Mean_Output['y_bar'])
    RMSE = metrics.mean_squared_error(Mean_Output['target'],Mean_Output['y_bar'])**.5
    print('NN Model\n Validation metrics (ensemble mean): \nr2 = ',
            np.round(R2,5),'\nRMSE = ',np.round(RMSE,5))
    T2 = time.time()
    print('Run Time:\n', np.round(T2 - T1,2),' Seconds')
        
    if config['RF_comp']==True:
        Run_RF(config,Data)

    return (RI,Mean_Output,full_out)

def Train_RF(config,Data):
    # Create a RF model for comparisson
    print('\nTraining RF Model')
    T1 = time.time()
    RF = RandomForestRegressor()
    RF.fit(Data['X'],Data['Y'])
    joblib.dump(RF, f"{config['Name']}/random_forest.joblib")
    T2 = time.time()
    print('Training Time:\n', np.round(T2 - T1,2),' Seconds')

def Run_RF(config,Data):
    # Run a RF model for comparisson
    T1 = time.time()
    RF = joblib.load(f"{config['Name']}/random_forest.joblib")
    y_pred = np.array([tree.predict(Data['X']) for tree in RF.estimators_])

    R2 = metrics.r2_score(Data['Y'],y_pred.mean(axis=0))
    RMSE = metrics.mean_squared_error(Data['Y'],y_pred.mean(axis=0))
    
    print('\n\nRF Model \nValidation metrics: \nr2 = ',np.round(R2,5),'\nRMSE = ',np.round(RMSE,5))
    
    RI = pd.DataFrame(index=config['inputs'],data = {'RI':RF.feature_importances_})
        
    N = y_pred.shape[0]
    if N > 1:
        t_score = stats.t.ppf(0.95,N)
    else:
        t_score = np.nan

    Mean_Output = pd.DataFrame(data = {
            'target':Data['Y'],
            'y_bar':y_pred.mean(axis=0).flatten(),
            'y_CI95':y_pred.std(axis=0).flatten()/(N)**.5*t_score
        })

    T2 = time.time()
    print('Run Time:\n', np.round(T2 - T1,2),' Seconds')
    return (Mean_Output,RI)

## Created by Dr. June Skeeter

## To DO
#  - save config & training data?
#  - expand to handle multiple targets?

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

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
    with open(f"{config['Name']}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)
        model_architecture = model_from_json(json.dumps(architecture))

    for i in range(config['N_models']):
        model = keras.models.clone_model(model_architecture)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                f"{config['Name']}/model_weights"+str(i)+".h5",
                save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=0),
        ]

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )

        model.fit(
            Data['X_train'],Data['Y_train'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=callbacks,
            validation_split=0.2,
            verbose=0,
        )

        tf.keras.backend.clear_session()

def get_SSD(dy_dx):
    N =  dy_dx.shape[0]
    SSD = ((dy_dx)**2).sum(axis=1).mean(axis=0)
    if N > 1:
        SSD_CI = ((dy_dx)**2).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
    else:
        SSD_CI = SSD*np.nan
    return(SSD,SSD_CI)

def run_Model(config,Data):

    with open(f"{config['Name']}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)

    full_out = {
        'y_pred':[],
        'dy_dx':[],
    }
    
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
                if not tf.is_tensor(Data['X_eval']):
                    x = tf.convert_to_tensor(Data['X_eval'])
                else:
                    x = Data['X_eval']
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
            'target':Data['Y_eval'],
            'y_bar':full_out['y_pred'].mean(axis=0).flatten(),
            'y_CI95':full_out['y_pred'].std(axis=0).flatten()/(N)**.5*t_score
        })
    

    for i,xi in enumerate(config['inputs']):
        Mean_Output[f'{xi}']=Data['X_eval'][:,i]
        Mean_Output[f'dy_d{xi}']=full_out['dy_dx'].mean(axis=0)[:,i]
        Mean_Output[f'dy_d{xi}_CI95']=full_out['dy_dx'].std(axis=0)[:,i]/(N)**.5*t_score

    SSD,SSD_CI = get_SSD(full_out['dy_dx'])
    RI = pd.DataFrame(index=config['inputs'],
                    data = {'RI_bar':SSD/SSD.sum(),
                            'RI_CI95':SSD_CI/SSD.sum()}
                    )
        
    if "dy_dx_norm" in full_out:
        for i,xi in enumerate(config['inputs']):
            Mean_Output[f'dy_d{xi}_norm']=full_out['dy_dx_norm'].mean(axis=0)[:,i]
            Mean_Output[f'dy_d{xi}_norm_CI95']=full_out['dy_dx_norm'].std(axis=0)[:,i]/(N)**.5*t_score
        SSD,SSD_CI = get_SSD(full_out['dy_dx_norm'])
        RI['RI_norm_bar']=SSD/SSD.sum()
        RI['RI_norm_CI95']=SSD_CI/SSD.sum()
        
    return (RI,Mean_Output,full_out)

            
# def make_SHL_model(input,hidden_nodes=64,dirname='SHL_NN_Model',activation='relu',
#                    Norm=False,m=None,v=None):
#     clean_dir(dirname)

#     input_shape = input.shape[-1]
#     input_layer = keras.layers.Input(input_shape)
#     if Norm == True:
#         norm = keras.layers.Normalization(mean=m,variance=v)
#         # norm.adapt(input)
#         x = norm(input_layer)
#     else:
#         x = input_layer
#     dense1 = keras.layers.Dense(
#                                 hidden_nodes,
#                                 activation=activation,
#                                 kernel_initializer="glorot_uniform",
#                                 bias_initializer="zeros"
#                                 )(x)
#     output_layer = keras.layers.Dense(1)(dense1)
#     model = keras.models.Model(inputs=input_layer, outputs=output_layer,name=dirname)
#     print(model.summary())
#     model_json = model.to_json()
#     with open(f"{dirname}/model_architecture.json", "w") as json_file:
#         json_file.write(model_json)

        
# def train_model_archive(x,y,dirname='SHL_NN_Model',Sub_Models=1,epochs = 250,batch_size = 32): 

#     with open(f"{dirname}/model_architecture.json", 'r') as json_file:
#         architecture = json.load(json_file)
#         model_architecture = model_from_json(json.dumps(architecture))

#     for i in range(Sub_Models):
#         model = keras.models.clone_model(model_architecture)
#         callbacks = [
#             keras.callbacks.ModelCheckpoint(
#                 f"{dirname}/model_weights"+str(i)+".h5", save_best_only=True, monitor="val_loss"
#             ),
#             keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=0),
#         ]

#         model.compile(
#             optimizer="adam",
#             loss="mean_squared_error",
#         )

#         model.fit(
#             x,
#             y,
#             batch_size=batch_size,
#             epochs=epochs,
#             callbacks=callbacks,
#             validation_split=0.2,
#             verbose=0,
#         )

#         tf.keras.backend.clear_session()

# def predict_Model(x,x_raw=None,target=None,dirname='SHL_NN_Model',Norm=True):

#     with open(f"{dirname}/model_architecture.json", 'r') as json_file:
#         architecture = json.load(json_file)
#     pred = []
#     deriv = []
#     for (_, _, filenames) in os.walk(dirname):
#         for f in filenames:
#             if f.split('.')[-1]=='h5':
#                 loaded_model = model_from_json(json.dumps(architecture))
#                 loaded_model.load_weights(f"{dirname}/{f}")
#                 loaded_model.compile(loss='mean_squared_error', optimizer='adam')
#                 if not tf.is_tensor(x):
#                     x = tf.convert_to_tensor(x)
#                 with tf.GradientTape(persistent=True) as tape:
#                     tape.watch(x)
#                     est = loaded_model(x)
#                 pred.append(est)
#                 deriv.append(tape.gradient(est,x))
#     if target is None:
#         return(pred,deriv)
#     else:
#         y = []
#         dy_dx = []
#         for i in range(len(pred)):
#             y.append(pred[i].numpy())
#             dy_dx.append(deriv[i].numpy())
            
#         y = np.array(y)
#         dy_dx = np.array(dy_dx)


#         N = y.shape[0]

#         out = pd.DataFrame(data = {
#             'target':target,
#             'y_bar':y.mean(axis=0).flatten(),
#             'y_CI95':y.std(axis=0).flatten()/(N)**.5*stats.t.ppf(0.95,N)
#         })

#         # Scale derivatives by the variance from the Normalization layer
#         # So they can be compared/ranked on the same scale
#         if 'Normalization' in json.dumps(architecture) and Norm == True:
#             for mod in architecture['config']['layers']:
#                 # print(m)
#                 if mod['class_name'] == 'Normalization':
#                     m = np.array(mod['config']['mean'])
#                     v = np.array(mod['config']['variance'])
#                     dy_dx_norm = (((dy_dx*v**.5)))
#                     break

#             SSD = (np.abs(dy_dx)).sum(axis=1).mean(axis=0)
#             SSD_CI = (np.abs(dy_dx)).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
#             SSD_norm = (np.abs(dy_dx_norm)).sum(axis=1).mean(axis=0)
#             SSD_CI_norm = (np.abs(dy_dx_norm)).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
#             RI = pd.DataFrame(data = {
#                 'RI':SSD/SSD.sum(),
#                 'RI_95':SSD_CI/SSD.sum(),
#                 'RI_norm':SSD_norm/SSD_norm.sum(),
#                 'RI_norm_95':SSD_CI_norm/SSD_norm.sum()
#             })

#         else:
#             SSD = (np.abs(dy_dx)).sum(axis=1).mean(axis=0)
#             SSD_CI = (np.abs(dy_dx)).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
#             RI = pd.DataFrame(data = {
#                 'RI':SSD/SSD.sum(),
#                 'RI_95':SSD_CI/SSD.sum()
#             })

#         if x_raw is not None:
#             x = x_raw.copy()
#         for i in range(x.shape[-1]):
#             out[f'x{i}']=x[:,i]
#             out[f'dy_dx{i}']=dy_dx.mean(axis=0)[:,i]
#             out[f'dy_dx{i}_CI95']=dy_dx.std(axis=0)[:,i]/(N)**.5*stats.t.ppf(0.95,N)
#             if 'Normalization' in json.dumps(architecture) and Norm == True:
#                 out[f'dy_dx{i}_norm']=dy_dx_norm.mean(axis=0)[:,i]
#                 out[f'dy_dx{i}_norm_CI95']=dy_dx_norm.std(axis=0)[:,i]/(N)**.5*stats.t.ppf(0.95,N)
        
#         return (RI,out)


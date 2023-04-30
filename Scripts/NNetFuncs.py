import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

def clean_dir(dirname):
    if os.path.isdir(dirname):
        for (_, _, filenames) in os.walk(dirname):
            for f in filenames:
                os.remove(dirname+'/'+f)
    else:
        os.mkdir(dirname)
            
def make_SHL_model(input_shape,hidden_nodes=64,dirname='SHL_NN_Model',activation='relu'):
    clean_dir(dirname)
    input_layer = keras.layers.Input(input_shape)
    dense1 = keras.layers.Dense(
                                hidden_nodes,
                                activation=activation,
                                kernel_initializer="glorot_uniform",
                                bias_initializer="zeros"
                                )(input_layer)
    output_layer = keras.layers.Dense(1)(dense1)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer,name=dirname)
    print(model.summary())
    model_json = model.to_json()
    with open(f"{dirname}/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

        
def train_model(x,y,dirname='SHL_NN_Model',Sub_Models=1,epochs = 250,batch_size = 32): 

    with open(f"{dirname}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)
        model_architecture = model_from_json(json.dumps(architecture))

    for i in range(Sub_Models):
        model = keras.models.clone_model(model_architecture)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                f"{dirname}/model_weights"+str(i)+".h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=0),
        ]

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )

        model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=0,
        )

        tf.keras.backend.clear_session()

        
def predict_Model(x,target=None,dirname='SHL_NN_Model'):

    with open(f"{dirname}/model_architecture.json", 'r') as json_file:
        architecture = json.load(json_file)
    pred = []
    deriv = []
    for (_, _, filenames) in os.walk(dirname):
        for f in filenames:
            if f.split('.')[-1]=='h5':
                loaded_model = model_from_json(json.dumps(architecture))
                loaded_model.load_weights(f"{dirname}/{f}")
                loaded_model.compile(loss='mean_squared_error', optimizer='adam')
                if not tf.is_tensor(x):
                    x = tf.convert_to_tensor(x)
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x)
                    est = loaded_model(x)
                pred.append(est)
                deriv.append(tape.gradient(est,x))
    if target is None:
        return(pred,deriv)
    else:
        # stand = StandardScaler()
        y = []
        dy_dx = []
        for i in range(len(pred)):
            y.append(pred[i].numpy())
            # stand.fit(deriv[i].numpy())
            # dy_dx.append(stand.transform(deriv[i].numpy()))
            dy_dx.append(deriv[i].numpy())
            
        y = np.array(y)
        dy_dx = np.array(dy_dx)


        N = y.shape[0]

        out = pd.DataFrame(data = {
            'target':target,
            'y_bar':y.mean(axis=0).flatten(),
            'y_CI95':y.std(axis=0).flatten()/(N)**.5*stats.t.ppf(0.95,N)
        })

        SSD = ((dy_dx)**2).sum(axis=1).mean(axis=0)
        SSD_CI = ((dy_dx)**2).sum(axis=1).std(axis=0)/(N)**.5*stats.t.ppf(0.95,N)
        
        RI = pd.DataFrame(data = {
            'RI':SSD/SSD.sum(),
            'RI_95':SSD_CI/SSD.sum()
        })
        
        for i in range(x.shape[-1]):
            out[f'x{i}']=x[:,i]
            out[f'dy_dx{i}']=dy_dx.mean(axis=0)[:,i]
            out[f'dy_dx{i}_CI95']=dy_dx.std(axis=0)[:,i]/(N)**.5*stats.t.ppf(0.95,N)
        
        return (RI,out)


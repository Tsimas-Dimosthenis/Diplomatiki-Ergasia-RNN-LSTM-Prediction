#Tsimas Dimosthenis A.M.: 4831
#Diplomatiki Ergasia
#Thema: Endosynedriaki Provlepsi Isotimias Synallagmatos me xrhsh RNN

import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter

import keras
from keras import metrics
from keras import regularizers
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

#config
file_name1='EU2USDER.csv'
file_name2='GBP2EUER.csv'
#epilogi arxeiou file_name1 i file_name2
FILE_NAME=file_name1
Epochs=65
U1=200
U2=200
architecture=1

#Orizoume custom class gia na mporesoume na ektypwsoume to training loss (mse) kai to validation loss
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
      
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        

#Read file
def read_file(fname):
    outframe = pd.read_csv(fname,header=0, parse_dates=[0],index_col=0, squeeze=True)
    print('Reading file:%s' %fname)
    return outframe

#Kanoume Pad ta data, reindex me tis imerominies pou leipoun
#Fill ta kena autwn me NaN, forward fill pass ta NaN me ta kontinotera non-NaN data pros ta katw
#Backwards fill pass mono gia tis teleutaies NaN times pou den antikatasta8ikan apo tin ffill
def padseries(df):
    r = pd.date_range(start='1-1-2015', end='31-12-2018')
    s = df.reindex(r).fillna(np.nan).reset_index()
    df=pd.Series(s['val'].values, index=s['index'])
    df=df.ffill()
    df=df.bfill()
    return df

#Metatropi se supervised problem me to prwto column na einai to input x (t+1)
#To 2o column na einai to y (t)
def supervision(df):
    df = pd.DataFrame(data=df)
    df = pd.concat([df.shift(1), df], axis=1)
    df=df.fillna(0)
    return df    

#Kanoume ta data stationary, metasximatizontas ta se lista diaforwn
def difference(df):
    diff = list()
    for i in range(1, len(df)):
        differ = df[i] - df[i - 1]
        diff.append(differ)
    return pd.Series(diff)

#Antistrofi enos antikeimenou mias listas diaforwn
def reverse_difference(original,prediction,index):
    r_difference=prediction+original[index]
    return r_difference

def alt_reverse_differences(original,predictions):
    reversed_5=list()
    reversed_prediction=reverse_difference(original,predictions[0],0)
    reversed_5.append(reversed_prediction)
    for j in range(1,6):
        reversed_prediction=reversed_prediction+predictions[j]
        reversed_5.append(reversed_prediction)
    return reversed_5


#Antistrofi tis listas diaforwn
def reverse_differences(original,predictions):
    reversed_predictions=list()
    #print(len(predictions))
    #print(len(original))
    for j in range(0,len(predictions)):
        reversed_prediction=reverse_difference(original,predictions[j],j)
        reversed_predictions.append(reversed_prediction)
    return reversed_predictions
    

#Elegxos an ta data einai stationary, visualization, statistics and null hypothesis 
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()
    
    plt.figure(figsize=(14,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Min-max scaling twn data, euros 0,1 , apo8ikeusi montelou gia reverse scaling meta
def scaling(df):
    scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
    df=df.reshape(len(df),1)
    scaler=scaler.fit(df)
    df_scaled=scaler.transform(df)
    df_scaled=df_scaled.reshape(len(df_scaled),1)
    return df_scaled, scaler

#Antistrofi tou min max scaling
def inverse_scaling(df, scaler):
    df_inv_scaled = scaler.inverse_transform(df)
    return df_inv_scaled

#Spasimo twn dedomenwn se train test sets 80-20 analogia
def split_to_train_and_test(df):
    df_values=df.values
    Y_scale=df_values[:,1]
    X_scale=df_values[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X_scale,Y_scale,test_size=.2, shuffle=False)
    return X_train, X_test, Y_train, Y_test

#Orismos montelou, ekpaideusi, dimiourgia custom history gia na e3agoume training/validation losses (mse metric)
#Work in progress i topologia kai ta hyperparameters
def LSTM_training(X_train,Y_train,epoch_size):
    X_train=X_train.reshape(X_train.shape[0],1,1)
    Y_train=Y_train.reshape(Y_train.shape[0],1)
    opt = keras.optimizers.SGD(lr=0.0008, momentum=0.9)
    model = Sequential()
    if architecture == 1 :
        model.add(LSTM(U1, input_shape=(1,1), batch_size=1,unit_forget_bias=True, stateful=True))
        model.add(Dropout(0.2))
    elif architecture == 2:
        model.add(LSTM(U2, input_shape=(1,1), batch_size=1,unit_forget_bias=True,return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
        model.add(LSTM(U2, batch_size=1,unit_forget_bias=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(Dense(Y_train.shape[1]))
    model.compile(loss='mse', optimizer=opt)
    print(model.summary())
    custom_hist = CustomHistory()
    custom_hist.init()

    #To montelo mas einai stateful opote sto telos ka8e epoch kanoume manual reset ta hidden states
    for i in range(epoch_size):
        print("Epochs : " + str(i))
        hist=model.fit(X_train,Y_train,batch_size=1,epochs=1,validation_split=0.2, verbose=2,shuffle=False,callbacks=[custom_hist])
        model.reset_states()
    print('Model training complete')
    return custom_hist,model
    
#Prediction 1-1    
def LSTM_prediction(X_input, model):
    X_input=X_input.reshape(1,1,1)
    prediction=model.predict(X_input, batch_size=1)
    return prediction

#Predictions gia ka8e input, reset ta states, epanalipsi tis parapanw diadikasias, 
def LSTM_mpredictions(X_input,model):
    predictions=list()
    for i in range(X_input.shape[0]):
        X=X_input[i]
        prediction=LSTM_prediction(X,model)
        predictions.append(prediction) 
        model.reset_states()       
    return predictions

def predict_five(X_input,model):
    predict5=list()
    X=X_input[0]
    prediction=LSTM_prediction(X,model)
    predict5.append(prediction)
    for i in range(5):
        prediction=LSTM_prediction(prediction,model)
        predict5.append(prediction)
    return predict5

#Plot provlepsewn vs pragmatikwn dedomenwn
def pred_print(df):
    df=df.to_numpy()
    fig, ax = plt.subplots()
    ax.plot(range(1,294),df[:,0], label = 'Real')
    ax.plot(range(1,294),df[:,1], 'r.', label = 'Predicted')
    ax.legend()
    plt.show()

def alt_evaluate_predictions(df):
    RMSE=list()
    df=df.to_numpy()
    df=df[~np.isnan(df).any(axis=1)]
    for i in range(0,5):
        print(df[i,:])
        error=math.sqrt((df[i,0] - df[i,1])**2)
        RMSE.append(error)
        print("RMSE for t+" +str(i+1)+ "= %.4f" %RMSE[i])
    #RMSE plot
    
    fig, loss_ax = plt.subplots()
    loss_ax.plot(range(1,6),RMSE[:], 'y', label = 'RMSE')
    loss_ax.set_xlabel('timestep t+i')
    loss_ax.set_ylabel('RMSE')
    loss_ax.legend(loc = 'upper left')
    plt.show()

#Plot tou training loss kai tou validation loss (metric mse)
def train_vall_loss_print(custom_hist):
    fig, loss_ax = plt.subplots()
    loss_ax.plot(custom_hist.train_loss, 'y', label = 'train loss')
    loss_ax.plot(custom_hist.val_loss, 'r', label = 'val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc = 'upper left')
    plt.show()


#Custom evaluation tou y predicted vs tou pragmatikou y me rmse
#Ka8arizoume prwta to arxeio ta rows pou periexoun NaN syni8ws ta teleutaia 1~6        
def evaluate_predictions(df):
    df=df.to_numpy()
    df=df[~np.isnan(df).any(axis=1)]
    RMSE=math.sqrt(mean_squared_error(df[:,0], df[:,1]))
    print("RMSE= %.4f" %RMSE)



#Read file & pad missing dates and values
df=read_file(FILE_NAME)
df=padseries(df)

#An 8eloume na eleg3oume ta input dedomena mas gia stationarity
#test_stationarity(df) 

#Kratame sto original ta input data mas gia mellontiki xrhsh
original=df

#Metatropi twn data se lista diaforwn
df=difference(df)

#An 8eloume na eleg3oume ta tropopoiimena dedomena diaforas mas gia stationarity
#test_stationarity(df)

#Metatropi twn data apo 1 column se 2, me to prwto column na einai to x kai to deutero to y
df=supervision(df)

#Split twn data se train kai test sets 80-20
X_train, X_test, Y_train, Y_test = split_to_train_and_test(df)

#Scaling twn data me min max scalar sto range 0,1
X_train,scalerXtrain=scaling(X_train)
Y_train,scalerYtrain=scaling(Y_train)
X_test, scalerXtest=scaling(X_test)

#Dhmiourgia kai training tou montelou mas gia osa epoch 8eloume-> Epoch number sto config pio panw
custom_hist, model=LSTM_training(X_train,Y_train,Epochs)

#Xtizoume to state twn LSTM prin 3ekinisoume forecasts
X_train=X_train.reshape(X_train.shape[0],1,1)
model.predict(X_train, batch_size=1)

#Plot tou training loss kai tou validation loss (metric mse)
train_vall_loss_print(custom_hist)

#Dhmiourgia predictions
#predictions=LSTM_mpredictions(X_test,model)
predictions=predict_five(X_test,model)

#Concat twn predictions se ena array sto idio column
predictions=np.concatenate(predictions, axis=0)
#Convert to pandas dataframe
predictions=pd.DataFrame(predictions)

#Antistrofi tou min max scaling sta predictions
predictions=inverse_scaling(predictions,scalerXtest)
print(predictions)

#Dhmiourgia dedomenwn gia ypologismo tou RMSE
#Ta dedomena auta antistoixoun sto Y test kommati (to teleutaio 20% tou Y), xwris na einai scaled/differenced omws
originalYtest = original[int(len(original) * .8) : int(len(original))]

#Antistrofi tis listas diaforwn, i opoia einai i morfi pou vgikan ta predictions
final=alt_reverse_differences(originalYtest,predictions)
print(len(final))

#Morfopoiisi final predictions kai Ytest gia na ta kanoume plot
final=np.concatenate(final, axis=0)
final=pd.DataFrame(final)
originalYtest=pd.DataFrame(originalYtest)
originalYtest=originalYtest.reset_index(drop=True)

#Concat ston idio pinaka, kai ta 2 antiproswpeuoun to Y (real vs predicted) tin idia xroniki stigmi den kanoume kapoio shift
df = pd.concat([originalYtest, final], axis=1)

#Plot tou Ytest kai tis provlepsis tin antistoixi xroniki stigmi
pred_print(df)
alt_evaluate_predictions(df)
#Ektypwsi RMSE
#WIP xrhsh tou RMSE gia e3antlhtiko hyperparameter optimization mesw framework
#evaluate_predictions(df)
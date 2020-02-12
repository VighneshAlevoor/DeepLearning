import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Preprocessing

dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#creating a data structure with time steps 60 and 1 output

x_train=[]
y_train=[]
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i,0])  # one column and 60 rows of past data(ex: 0 to 59th day data to predict 60th value)
    y_train.append(training_set_scaled[i,0]) # predict is i for previous 60 values
x_train,y_train=np.array(x_train), np.array(y_train)

#Reshaping (we can add indicators here if we know any parameter which effects stock rate) also 3d shaping for future compatibility
#we just took value as 1 
#check keras doc for values to input
x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Building RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
#no need to give full input shape as x_train,(x_train.shape[0],x_train.shape[1],1 : It takes first one automatic
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#output layer

regressor.add(Dense(units=1))

#Compile RNN:
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fit RNN
regressor.fit(x_train,y_train,epochs=100,batch_size=32)

regressor.save('C:\\Users\\TEMP.VIGHNESH-PC.001\\Desktop\\ML\\udemy-kiril\\Deep learning\\Recurrent_Neural_Networks\\RNN_model.h5')

#making predictions
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

#getting predicted values for 2017
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
#len(dataset_total - 1278 till jan month lst day
#len(dataset_test)-subtract 20 - gives Jan 3
# subtract 60 gives 3 months before day as we trained for 60 days our model 
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#plot
plt.plot(real_stock_price, color='red', label='real_stock_price')
plt.plot(predicted_stock_price, color='blue', label='predicted_stock_price')
plt.title('Google stock prediction')
plt.xlabel('Time')
plt.ylabel('Stock value')
plt.legend()
plt.show()
















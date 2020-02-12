import pandas as pd
import numpy as np

df=pd.read_csv("Credit_Card_Applications.csv")

#class indicates whether card issued or not
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
x=sc.fit_transform(x)

#Training the som
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

#visualizing
#som.distance_map gives mean inter neuron distances, matrix of ditnace of winning nodes
#Distance_map Returns the distance map of the weights. Each cell is the normalised sum of the
#distances between a neuron and its neighbours.
#Transpose needed to take right order for pcolor function
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
#colorbar for legends
#outlying winning nodes are in white color and they are farthest and hence the potential cheater
colorbar()

#markers to show: red - didnt get approval. green- got approval
#if green and falls in white box - need to consider them as potential cheaters
markers=['o','s'] #circle -o, squarec-s
colors=['r','g']

#for index i(suppose first row index) takes all values of row first (j)
#j is all the vectors at different iterations
for i,j in enumerate(x):
    w=som.winner(j) #gives winning node of customer j
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]], #if approved marks green sqaure if not then red circle, as per y data
         markeredgecolor=colors[y[i]], #markers[y[i]] means if not appoved y[i]=0 then mrkers[0]= circle  
         markerfacecolor='None', #coloronly border since we can have many values in one cell
         markersize=10,
         markeredgewidth=2)  
show()
    
#finding the frauds
mappings=som.win_map(x)   #gives winning node having how many customers
frauds=np.concatenate((mappings[(4,5)],mappings[(2,2)],mappings[(2,3)]),axis=0)
frauds=sc.inverse_transform(frauds)

#going from unsupervised to supervise

#create matrix of features
customers=df.iloc[:,1:].values

#create dependent variable , col is_fraud
#we have customers who fraud, assign those customers 1 and others 0

is_fraud=np.zeros(len(df))
for i in range(len(df)):
    if df.iloc[i,0] in frauds:
        is_fraud[i]=1
        
        
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
customers=sc.fit_transform(customers)
 
#ANN       
from keras.models import Sequential
from keras.layers import Dense 

classifier=Sequential()
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))
classifier.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

#predict probabilities
y_pred=classifier.predict(customers)
y_pred=np.concatenate((df.iloc[:,0:1].values, y_pred),axis=1)
y_pred=y_pred[y_pred[:,1].argsort()]























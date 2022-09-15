from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import os
import json

# Data Manupulation
import numpy as np
import pandas as pd
import datetime

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Data fetching
from pandas_datareader import data as pdr
import yfinance as yf

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read the data from Yahoo
df= pdr.get_data_yahoo('PSEI.PS', '2020-01-01', '2022-12-31')

## Actual Data
# print (df)

df = df.dropna()
df = df[['Open', 'High', 'Low','Close']]
df.head()

# Predictor variables
df['Open-Close']= df.Open -df.Close
df['High-Low']  = df.High - df.Low
df =df.dropna() # Remove values with NaN
X= df[['Open-Close','High-Low']]
X.head()

# Target variable
Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)

# Splitting the dataset
split_percentage = 0.4 # Use 60% of data for training, 40% for testing
split = int(split_percentage*len(df))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

# Instantiate KNN learning model(k=15)
knn = KNeighborsClassifier(n_neighbors=15)

# fit the model
knn.fit(X_train, Y_train)

# Accuracy Score
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

# print ('Train_data Accuracy: %.2f' %accuracy_train)
# print ('Test_data Accuracy: %.2f' %accuracy_test)

# Predicted Signal
df['Predicted_Signal'] = knn.predict(X)

# PSEi Cumulative Movement
df['PSEi_movement'] = np.log(df['Close']/df['Close'].shift(1))
df['Cumulative_PSEi_returns'] = df[split:]['PSEi_movement'].cumsum()*100

# Cumulative Strategy Returns 
df['Startegy_movement'] = df['PSEi_movement']* df['Predicted_Signal'].shift(1)
df['Cumulative_Strategy_returns'] = df[split:]['Startegy_movement'].cumsum()*100

#print (df)

# Calculate Sharpe reatio
Std = df['Cumulative_Strategy_returns'].std()
Sharpe = (df['Cumulative_Strategy_returns']-df['Cumulative_PSEi_returns'])/Std
Sharpe = Sharpe.mean()
# print('Sharpe ratio: %.2f'%Sharpe )

append = {
  "sharpe_ratio":Sharpe,
  "test_accuracy":accuracy_test,
  "train_accuracy":accuracy_train  
}

#printing values
print ('Train_data Accuracy: %.2f' %accuracy_train)
print ('Test_data Accuracy: %.2f' %accuracy_test)
print('Sharpe ratio: %.2f'%Sharpe )

#Plot the results to visualize the performance
##plt.figure(figsize=(10,5))
##plt.plot(df['Cumulative_PSEi_returns'], color='r',label = 'PSEi Movement')
##plt.plot(df['Cumulative_Strategy_returns'], color='g', label = 'KNN Algorithm Movement')
##plt.legend()
##plt.show()


#parsing to json to be ready to deploy to server 
##result = df.to_json(orient="table")
##parsed = json.loads(result)
##parsed.update(append)
##json.dumps(parsed, indent=4)  
##
##@app.route("/")
##def hello_world():
##  return json.dumps(parsed, indent=4) ;
##
##port = int(os.environ.get('PORT', 5000))
##app.run(host='0.0.0.0', port=port, debug=True)

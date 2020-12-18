#formula of a straight line, y = mx + b
#regression find out what m and b is
#stock prices program
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key = "a1Ep9sLFKx8XZcoGbDk-"

df = quandl.get_table("WIKI/PRICES")
df = df[['open','adj_high','adj_low','adj_close','adj_volume']]
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['open']) / df['open'] * 100.0

df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)



x = np.array(df.drop(['label'],1))
x = x[:-forecast_out]
x = preprocessing.scale(x)
df.dropna(inplace=True)
y = np.array(df['label'])
y= np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#classifier
clf = LinearRegression(n_jobs=10)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy)
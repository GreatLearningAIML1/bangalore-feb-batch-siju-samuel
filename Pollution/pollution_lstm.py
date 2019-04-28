from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    print("*******n_vars = ", n_vars, " n_in=", n_in, "n_out=", n_out)
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    print("NAMES = ", names)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
print(dataset.head(3))
print("dataset.shape =", dataset.shape)
print("dataset.info =", dataset.info())

print("INFO  OK")

values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 3
n_features = 8
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)
print("reframed.info =", reframed.info())
print(reframed.head(3))

# split into train and test sets
values = reframed.values
n_train_hours = int(len(values) *0.7) #365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
print("n_train_hours = ", n_train_hours, " n_hours=", n_hours, " n_features=", n_features, " n_obs=",n_obs)
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("Shapes train_X =", train_X.shape, " train_y = ", train_y.shape, " test_X = ", test_X.shape, " test_y = ", test_y.shape)

#from sklearn.model_selection import train_test_split
#train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=.3, random_state=42)



# design network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(50,  return_sequences=True))
model.add(LSTM(50))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

import numpy as np

def rmsle2(y0, y):
    print(type(y))
    print(type(y0))
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))

def rmsle3(real, predicted):
    sum=0.0
    print("Total predicted = ", len(predicted))
    for x in range(len(predicted)):
        print("Values are predicted=", predicted[x], " real=", real[x], " Diff = ", (predicted[x] - real[x]))
        if predicted[x]<0 or real[x]<0: #check for negative values
            print("Have -ve values, predicted=", predicted[x], " real=", real[x])
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle1(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def rmsle(y_true, y_pred):
    np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

RMSLE = rmsle3(inv_y, inv_yhat)
print('Test RMSLE: %.3f' % RMSLE)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
#print("Accuracy: ", accuracy_score(inv_y, inv_yhat))

# print precision, recall, F1-score per each class/tag
#print("classification_report: ",classification_report(inv_y, inv_yhat))

# print confusion matrix, check documentation for sorting rows/columns
#print("confusion_matrix: ", confusion_matrix(inv_y, inv_yhat))

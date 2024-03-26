"""
pip install flask-ngrok
"""
import os
import pickle
import pandas
import warnings

from flask import Flask, request
from flask_ngrok import run_with_ngrok
import requests

from matplotlib import pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression,ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 100)
pandas.set_option('mode.chained_assignment', None)
pandas.set_option('compute.use_numexpr', False)

months_word = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')

weeks_word = ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
weeks_num = (1,2,3,4,5,6,7)



months_num = tuple(range(1,13))

from icecream import ic
filename = 'forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
ic(pandas.isnull(df).sum())
ic(df.shape)
ic(df.dtypes)
ic(df.head()) # top 5 rows
ic(df.describe())
df.month.replace(months_word, months_num, inplace=True)
df.day.replace(weeks_word, weeks_num, inplace=True)
ic(df.head()) # top 5 rows
ic(df.corr(method='pearson'))

#df.hist(sharex=False, sharey=False, xlabelsize=15, ylabelsize=15, color='orange', figsize=(15,15))
#plt.suptitle('Histagrams', y=1.00, fontweight='bold', fontsize=40)
#plt.show()

# probability density function(概率密度函数,PDF)
#df.plot(kind='density', subplots=True, layout=(7,2), sharex=False, fontsize=16, figsize=(15,15))
#plt.suptitle('PDF', y=1.00, fontweight='bold', fontsize=40)
#plt.show()

#df.plot(kind='box', subplots=True, layout=(4,4), sharex=False, fontsize=16, figsize=(15,15))
#plt.suptitle('Box and Whisker', y=1.00, fontweight='bold', fontsize=40)
#plt.show()

#Axes = scatter_matrix(df, figsize=(15,15))
#plt.suptitle('Scatter Matrix', y=1.00, fontweight='bold', fontsize=30)
#plt.rcParams['axes.labelsize'] = 15
#[plt.setp(item.yaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
#[plt.setp(item.xaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
#plt.show()

plt.figure(figsize=(11,11))
plt.style.use('default')
sns.heatmap(df.corr(), annot=True)

x, y  = np.arange(10).reshape((5,2)), range(5)
ic(x)
ic(list(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)
ic(x_train)
ic(x_test)
ic(y_train)
ic(y_test)

dataset = range(16)
KFCrossValidator = KFold(n_splits=4, shuffle=False)
KFdataset = KFCrossValidator.split(dataset)
print("{} {:^61} {}".format("Round", "Training set", "Testing set"))

for iteration, data in enumerate(KFdataset, start=1):
    print(f"{iteration:^9} {data[0]} {str(data[1]):^45}")

arr = df.values
x = arr[:,0:12]
y = arr[:,12]
ic(x)
ic(y)
#os._exit(0)

max_error_scoring = 'max_error'
neg_mean_absolute_error_scoring = 'neg_mean_absolute_error'
r2_scoring = 'r2'
neg_mean_squared_error_scoring = 'neg_mean_squared_error'

models = []
models.append((['LR', LinearRegression()]))
models.append((['LASSO', Lasso()]))
models.append((['EN', ElasticNet()]))
models.append((['Ridge', Ridge()]))
models.append((['KNN', KNeighborsRegressor()]))
models.append((['CART', DecisionTreeRegressor()]))
models.append((['SVR', SVR()]))


for name, model in models:
    #kfold = KFold(n_splits=10, random_stat=7)
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, x, y, cv=kfold, scoring=max_error_scoring)
    cv_results1 = cross_val_score(model, x, y, cv=kfold, scoring=neg_mean_absolute_error_scoring)
    cv_results2 = cross_val_score(model, x, y, cv=kfold, scoring=r2_scoring)
    cv_results3 = cross_val_score(model, x, y, cv=kfold, scoring=neg_mean_squared_error_scoring)
    msg = '%s max error: %f, mean absolute error: %f, r2:%f, mean squared error: %f' %(name, cv_results.mean(), cv_results1.mean(), cv_results2.mean(), -cv_results3.mean())
    ic(msg)

# os._exit(0)
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.20, random_state=1, shuffle=True)
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

predictions = lasso_model.predict(x_test)
#print(predict)
pickle.dump(lasso_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

#app = Flask(__name__)
#run_with_ngrok(app)
#
#@app.route('/predict', method=['POST'])
#def home():
#    x = int(request.arg.get('x', ''))
#    y = int(request.arg.get('y', ''))
#    month = int(request.arg.get('month', ''))
#    day = int(request.arg.get('day', ''))
#    FFMC = int(request.arg.get('FFMC', ''))
#    DMC = int(request.arg.get('DMC', ''))
#    DC = int(request.arg.get('DC', ''))
#    ISI = int(request.arg.get('ISI', ''))
#    temp = int(request.arg.get('temp', ''))
#    RH = int(request.arg.get('RH', ''))
#    wind = int(request.arg.get('wind', ''))
#    rain = int(request.arg.get('rain', ''))
#    prediction = lasso_model.predit([[x, y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])
#    print('*'*100)
#    print(prediction)
#    return 'Prediction is ' + str(prediction[0])
#
#app.run()



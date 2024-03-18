import pandas
import numpy
import warnings
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 100)
from icecream import ic
filename = 'forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
ic(pandas.isnull(df).sum())
ic(df.shape)
ic(df.dtypes)
ic(df.head()) # top 5 rows
ic(df.describe())
ic(df.corr(method='pearson'))



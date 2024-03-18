import pandas
import warnings
from icecream import ic
filename = 'forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
ic(pandas.isnull(df).sum())



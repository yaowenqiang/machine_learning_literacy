import pandas
import numpy
import warnings
from matplotlib import pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
import seaborn as sns
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



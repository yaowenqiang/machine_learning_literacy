# Machine Learning

Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead.

## Some Applications of Machine Learning

+ Image Recognition
+ Malware detection
+ Voice Recognition

## Traditional Programming vs Machine Learning

Traditional Programming |  machine Learning  |
-------  |  ------- |
We have Full Understanding of the domain    | We have Vague Understanding of the domain
solution rules are static | Solution rule are dynamic
Does not require historical data | Does require historical data
Straight forward and obvious  | Vague and tricky to understand


## Machine learning Algorithms Types

+ Supervised Learning
+ Unsupervised Learning
+ Reinforcement Learning

### Supervised Learning Types

+ Regression
+ Classification

### Unsupervised Learning Types

+ Clustering
+ Association


### Learning Types in a Nutshell

------- | Supervised  | Unsupervised | Reinforcement  | 
Objective  | Predict future values or categories  | Organize data based on underlying structure  | Adapt based on the rewards and data from the surrounding environment | 
Learning Source | Output dataset | input data patterns | Environment state and rewards | 


## Learning Modes

+ Batch Learning
+ Online Learning

## Machine Learning Pipeline


### Problem Definition

Five question ML can answer

+ Is this A or B ? Will this customer by or not ?
+ Is this wired 
+ How much - or how many ? How many items well i see in upcoming quoter? 
+ How is this organized ? What are the different customer categories do I have ?
+ What should i do next?

### Data sourcing

+ Several data sources may exist
  + RESTful Endpoints
  + File integration
  + SOAP Endpoints
  + SQL Table
  + Sensors with proprietary format

### Data preparing

+ Several data preparation actions
  + Dealing with missing data
  + Dropping unnecessary attributes
  + Detecting outliers
  + Etc.

### data Segregation

Data is segregated to

+ Training set
+ Validation set

### Model Training

Machine Learning algorithms adjusts
usually on-shelf recipe
May need trying several algorithms

### Model evaluation

Examining model performance using validation data
Different performance measures based on the algorithm type+

### Model deployment

Making model useful for business
Deployed Machine Learning model can take several formats

### model monitoring



+ mean
+ median
+ percentiles

Standard Deviation(标准差)

+ Consider all items
+ Considers data distribution
+ Harder to calculate


## Bivariate Measures(双变量测量)

## Correlation(相关性)

+  Positive Correlation
+  Negative  Correlation
+  No  Correlation

## Correlation Policy

Correlation Fallacy(相关性谬误)

Correlation does not imply casuation
("With this, therefore because of this" fallacy)


## Correlation(相关性)

## Density Graphs(密度曲线图)

## Common Distribution Types

+ Normal Distribution(正太分布)
+ Skewed Distribution(偏态分布)
+ Exponential Distribution(指数分布)

## Why Histograms and Density Graphs ? 

+ Detecting impossible values
+ Identifying the shape of the data
+ Detecting errors and mistakes in the data

interquartile range(四分位距)

## Box and Whisker Plot(盒形图与箱须图)

## Scatter Plot(散点图)


euclidean distance(欧几里得距离)

设有两点 P 和 Q，它们的坐标分别为 P=(x1, y1) 和 Q=(x2, y2)，那么 P 和 Q 之间的欧几里得距离 D 可以用以下公式计算：

 D = sqrt((x2-x1)^2 + (y2-y1)^2)


## K-Means clustering and Data Scale


Euclidean distance is affected by the magnitudes(量级) of the input dataset, and since conversion units(e.g, inch to cm) changes the magnitude, Euclidean Distance results will change


## Eliminating Scale Effect

### Data Scaling (数据绽放)

+ Standardization - Removing the mean and scaling to unit variance
+ MinMax Scaling - Rescaling all attributes to range between zero and one 
+ Normalization Scaling - Rescaling each observation(row) to unit value


As a rule of thumb, always scale your data when the underlying algorithm calculates distance

## Data Segregation(数据隔离)

Training and testing on the same set can result int overfitting

## Underfitting and Fitting (欠拟合与过拟合)

## Data Segregation Techniques

+ Train/Test Split
  + Training set 70% to 80%
  + Test set 20% to 30%
+ K-Fold Cross Validation

# Model Training and Evaluation


## What's Model Training

+ All Machine learning algorithms use one principle
+ Three types of Machine Learning algorithms(generally)
  + Supervised Learning
  + Unsupervised Learning
  + Reinforcement Learning

## Types of Supervised Learning Algorithms

+ Regression - for continuous values
+ Classification - For discrete(categorical) values
+


In the context of Supervised Learning: Machine Learning training is the process of learning an ML algorithm how to find patterns in the input data so that they correspond to the target, resulting a machine learning  model


## Foundational Concepts

+ Line slope(直线的斜率)
+ Derivative (导数)
+ instantaneous slope (瞬时斜率)
+

If we want to get the minimum of a function, then we calculate its derivative when it is equal to zero!

## Linear Regression Algorithms

### Bias-Variance Tradeoff(偏差-方差窘境)

## Regularization (正则化)

## Linear Regression Regularization

+ Ridge Regression
+ Lasso Regression
+ Elastic Net Regressions (Combines techniques from Ridge and Lasso)
+ K-neighbors Regression
  + Simple algorithm
  + Relies on distance measurement
  + Requires standardization
+ Support Vector Regression (SVR)
  + Used for both regression and classification (SVM)
  + Similar to linear regression
+ Decision Tree Regressor
  + Used for both regression and classification
  + Reaches the answer by structuring the data in three leaves


## MOdel Evaluation

### Regression Models Evalution Metrics

+ Max Error
  + Captures the worst case
  + How much we can tolerate
+ Mean Absolute Error  (平均绝对误差)
  + Average of absolute errors
+ Mean Squared Error (均方误差)
  + Average of squared errors
+ R squared 
+ Others


MAE vs. MSE 

MAE  |  MSE | 
---- |  ---- |
Removes negative signs by taking absolute value  | Removes negative signs by squaring the values
More robust to outliers |  Better when we want to penallze outliers

R^2 (Coefficient of Determination) (决定系数)

Tell us how much percentage of the data is explained by the relationship but no direction

Median absolute error
Mean squared log error



# Problem : Trying to build a model that will predict housing price based on existing features.
import numpy as np
import pandas as pd

df = pd.read_csv("D:\ML_Python\LinearRegression\HousingData.csv")
df.head()
df.info()
df.describe()
df.columns

# check the distribution of price column:
import seaborn as sns
sns.distplot(df["Price"])

# we can drop address column as it's not numeric.
df.columns
# grab only required columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = df["Price"] #target variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101) #random_state is used when you want random split to be fixed (e.g your random split of data equals to my random split).
X_test.info()
X_train.info()

from sklearn.linear_model import LinearRegression
lm = LinearRegression() # create a linear regression object/ model
lm.fit(X_train,y_train) # fit that model into a training set
lm.intercept_ # than find the intercept
lm.coef_ # and Coefficient

X_train.columns
cdf = pd.DataFrame(data=lm.coef_,index=X_train.columns,columns=["Coeff"])
cdf
# How to intepret price with these coefficients
# example
# with one unit increase in Avg. Area Income there will be 21.53 dollar price increase.

# Real time dataset which is already present in skleanr
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys() # boston is a dictionary wtih bunch of information on it.
boston["DESCR"]
boston["data"]



###### Prediction from the test set: #####
prediction = lm.predict(X_test)
# actual price is :
y_test
# we can visualize the predicted and actual price by scatter plot:
import matplotlib.pyplot as plt
plt.scatter(y_test,prediction) # linearly scattered , which indicates your model is linear.
# Residual plot:
sns.distplot(y_test-prediction) # Normally Distributed plot is very good sign that your model is correct.

# Regression Evaluation Metrics:MAE, MSE, RMSE
from sklearn import metrics
metrics.mean_absolute_error(y_test,prediction)
metrics.mean_squared_error(y_test,prediction)
np.sqrt(metrics.mean_squared_error(y_test,prediction))


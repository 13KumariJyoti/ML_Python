import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read Ecommerce data:
# It has Customer info:  such as Email, Address, and their color Avatar.
# Avg. Session Length: Average session of in-store style advice sessions.
# Time on App: Average time spent on App in minutes
# Time on Website: Average time spent on Website in minutes
# Length of Membership: How many years the customer has been a member.

customers = pd.read_csv("D:\ML_Python\LinearRegression\Ecommerce Customers.csv")
customers.head()
customers.info()
customers.describe()

# Jointplot to see the time spent on website and Time on App with yearly amount spent:
sns.jointplot(data=customers,x= "Time on Website",y ="Yearly Amount Spent")
sns.jointplot(data=customers,x= "Time on App",y ="Yearly Amount Spent")

# plot pair plot
sns.pairplot(customers) # column name: Length of Membership is having good relation with Yearly Amount Spent, so let's plot linearly:
sns.lmplot("Length of Membership","Yearly Amount Spent",data=customers)

# split the data into train and test:
customers.columns
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# Now train our model with training data:
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)  ## train lm on training data:
lm.coef_

# after fitting our model lets predict with test data:
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions) # scatter plot of Real test value and predicted values.
sns.distplot(y_test-predictions) #Residual Normally Distributed

# Evaluating Model by performance metrics: MAE, MSE, RMSE
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))

# Explained variance score i.e. R2 :measurement of how much variation your model explains.
metrics.explained_variance_score(y_test,predictions) # 99 percent variation is explained by model , which is very good .

# Interpret Coefficients :
Coeff = lm.coef_
pd.DataFrame(data= Coeff,index= X_train.columns,columns=["Coeff"])
# If you hold all the other features fixed ,with one unit increase in Time on App the Yearly Amount spent will increase to 38.6 unit by the customer.
# If you hold all the other features fixed ,with one unit increase in Time on Website the Yearly Amount spent will increase to 0.19 unit by the customer.

#  we can actually spend more in developing App as it's already giving good result or we can work more on Website to improve and get more benefits based on the cost association.


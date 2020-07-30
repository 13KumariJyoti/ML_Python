# In this project we will be working with a fake advertising data set,
# indicating whether or not a particular internet user clicked on an Advertisement.
# We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

import pandas as pd
ad_data = pd.read_csv("D:\ML_Python\LogisticRegression\Advertising.csv")
ad_data.head()
ad_data.info()
ad_data.describe()

# Exploratory data analysis:
# histogram of  age:
ad_data["Age"].plot.hist(bins = 35)
# JointPlot showing Area Income Vs Age:
import seaborn as sns
sns.jointplot(data=ad_data,x="Area Income",y="Age")
sns.jointplot(data=ad_data,y="Daily Internet Usage",x="Daily Time Spent on Site")

# missing values:
sns.heatmap(train.isnull()) # non of the column is having null value.

# let's drop the column which are not going to be useful:
ad_data.drop(["Ad Topic Line","City","Country","Timestamp"],axis=1,inplace=True)

# Split the data into train and test and let's train the model:
X = ad_data.drop("Clicked on Ad",axis=1)
y = ad_data["Clicked on Ad"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 101)

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X= X_train,y=y_train)

# prediction and evaluation :
predictions = logReg.predict(X_test)
# Classification report for model:
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

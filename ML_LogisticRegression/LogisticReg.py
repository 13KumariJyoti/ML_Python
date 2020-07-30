import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("D:\ML_Python\LogisticRegression\TitanicTrain.csv")
train.head()
# exploratory data analysis:
train.isnull()
train["Age"].isnull()
sns.heatmap(train.isnull()) # white strips are missing values in that particular column.
# visualize the target column : it's always good to do a countplot for classification problem:
sns.countplot(x = "Survived",data = train) # 0: not survived, 1: survived.
sns.countplot(x = "Survived",data = train,hue="Sex")

train["Age"].plot.hist(bins = 35)

# let's explore the data now :
train.info()
# nunmber of siblings and spouses on board:
sns.countplot(x="SibSp",data= train) # it shows that most people on board didn't have spouses or children on board.
train["Fare"].plot.hist()

# Part 2:
# treating missing value by filling with mean/Averge of that particular column:
# let's plot box plot between Pclass and age:
sns.boxplot(x="Pclass",y="Age",data=train)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train["Age"] = train[["Age","Pclass"]].apply(impute_age,axis=1)
sns.heatmap(train.isnull()) #all the missing values in Age column is now replaced with mean/Average.
# In Cabin column there are too many missing values so better to drop this column:
train.drop("Cabin",axis=1,inplace=True)
train.columns

# Deal with categorical Feature [adding dummy column]
pd.get_dummies(train["Sex"]) #this will give two column , but both columns are having relationship with each other
# e.g: if F = 0 than M = 1 , so to avoid this multicolinearity problem we will drop one column.
sex= pd.get_dummies(data=train["Sex"],drop_first=True)
embark = pd.get_dummies(data=train["Embarked"],drop_first=True)

# concatenate all these columns to our dataframe :
train = pd.concat([train,sex,Embarked],axis=1)

# we can drop the columns we are not going to use:
train.drop(["Sex","Embarked","Name","Ticket"],axis=1,inplace=True)
train.columns

# Since PassengerId is just an index column which is not actually helpful:
train.drop("PassengerId",axis=1,inplace=True)

# let's suppose this train data as full data and split it into train and test:
X = train.drop("Survived",axis=1)
y = train["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() #create a model
logmodel.fit(X= X_train,y = y_train) #fit a model
predictions = logmodel.predict(X_test) #predict a model

# let's evaluate our model for classifications:
from sklearn.metrics import classification_report  #to get True values , accuracy , precision etc.
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

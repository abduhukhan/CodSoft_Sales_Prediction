import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('advertising.csv')

#Check out the length of dataset which provides us the number of rows
print(len(data))
#Check out the head and names of columns of the dataset
print(data.head())
print(data.columns)
#Check out the info of dataset
print(data.info())
#Check out the statistical values of the dataset
print(data.describe())

#NULL VALUES
print(data.isna())
print(data.isna().sum())

#DATA VISUALIZATION USING SEABORN
sns.pairplot(data,hue="Sales")
plt.show()

#Extracting Dependent and Independent Variables
x=data[['TV','Radio','Newspaper']]
y=data[['Sales']]

#LOGISTIC REGRESSION
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

reg=LinearRegression()
reg.fit(x_train,y_train)
predict=reg.predict(x_test)

# REGRESSION METRICS
mse = mean_squared_error(y_test, predict)
r2 = r2_score(y_test, predict)

print("Mean Squared Error:", mse)
print("R-squared:", r2)



#load library
import pickle
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
np.random.seed(42)  #constant output

#load data
LoadData=pd.read_csv(r"E:\بايثون\P14-Part2-Regression\Section 6 - Simple Linear Regression\Python\Salary_Data.csv")
#load sample from data
print(LoadData.head())

# describe data 
print(LoadData.describe())

# show missing data
print(LoadData.info())

#lablel data
x=LoadData.drop("Salary",axis=1)# input
y=LoadData["Salary"] #output

# split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#scale data and choose model
model=make_pipeline(MinMaxScaler(),LinearRegression())
print(model)

#train model 
model.fit(x_train,y_train)

#best train ,best test
print(model.score(x_train , y_train))
print(model.score(x_test , y_test))

#draw model train
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,model.predict(x_train),color="blue")
plt.title("salary vs years experence")
plt.xlabel("years of expence ")
plt.ylabel("salary")
plt.show()

#draw model test
plt.scatter(x_test,y_test,color="black")
plt.plot(x_train,model.predict(x_train),color="red")
plt.title("salary vs years experence")
plt.xlabel("years of expence ")
plt.ylabel("salary")
plt.show()


#save model
pickle.dump(model,open(r"salary_data.pkl","wb"))
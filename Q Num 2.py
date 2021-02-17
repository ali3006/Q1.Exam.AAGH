import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
File = pd.read_csv("C:\\New folder\\Diabetes_Diagnosis.csv")
features = ['glucose_conc','insulin','bmi','age',]
X = File[features]
def fcam(diabetes):
    if diabetes == True:
        diabetes = '1'
    if diabetes == False:
        diabetes = '0'
    return diabetes
Y = File.diabetes.apply(fcam)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=100)
Z = LinearRegression()
Reg = Z.fit(X_train , Y_train)
print('Regression = ' , Reg.coef_)
#Row Like
RL = pd.concat([pd.DataFrame(X_train.columns) , pd.DataFrame(np.transpose(Reg.coef_))] , axis = 1)
print(RL)
Status_male = X[X['diabetes'] == '0'].describe()
Status_male.rename(columns = lambda X: X + '_1' , inplace = True)

Status_female = X[X['diabetes'] == '1'].describe()
Status_female.rename(columns = lambda X: X + '_0' , inplace = True)
Status = pd.concat([Status_male, Status_female] , axis = 1)
print(Status)



y = '((Ali Akbar Gholami))'
z = y.center(67,'0')
print(z)
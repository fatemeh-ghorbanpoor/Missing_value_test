#These codes will be written in an object oriented way
import numpy as np
import pandas as pd
#Visualize missing values
import missingno as msno
import matplotlib.pyplot as plt
#replacing nan values by mean
from sklearn.impute import SimpleImputer

data = pd.read_csv('H://NISCO/codes/missed_value.csv')
Data = pd.DataFrame(data=data)
Data.drop('Unnamed: 0', axis = 1 ,inplace = True)
#print(Data)

#Numerics all data and replaces it with NAN if there are no values
for i in Data.columns:
    Data[i] = pd.to_numeric(Data[i], errors = 'coerce')

#print(Data)
print(Data.columns)
#print(Data.info())

'''#Deletes the rows including NAN
#print(Data.dropna(axis = 0))'''

'''#Instead of NAN,it puts zero
#print(Data.fillna(0))'''

'''#Replace with the previous number-forward-filling-problem: first row 
print(Data.fillna(method = 'ffill'))'''

'''#backward-filling
Data.fillna(method='bfill', inplace=True)'''


'''#Replace with mean
i = SimpleImputer(missing_values = np.nan, strategy='mean')
i.fit(Data)
new_Data = i.transform(Data)
print(new_Data)'''

'''# To insert the mean value of each column into its missing rows:
Data.fillna(df.mean(numeric_only=True).round(1), inplace=True)'''

'''#Replace with median
i = SimpleImputer(missing_values = np.nan, strategy='median')
i.fit(Data)
new_Data = i.transform(Data)
print(new_Data)'''

'''# For median- another way
Data.fillna(df.median(numeric_only=True).round(1), inplace=True)'''

'''#Replace with most frequent
i = SimpleImputer(missing_values = np.nan, strategy='most frequent')
i.fit(Data)
new_Data = i.transform(Data)
print(new_Data)'''

'''#Fill Missing Data With interpolate()-
# Interpolate backwardly across the column:
Data.interpolate(method ='linear', limit_direction ='backward', inplace=True)

# Interpolate in forward order across the column:
Data.interpolate(method ='linear', limit_direction ='forward', inplace=True)'''



'''#Visualize missing values
msno.matrix(Data)
plt.show()
msno.heatmap(Data)
plt.show()
msno.bar(Data)
plt.show()'''



##  not completed ##
####### Linear Regression  ###########
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


independent_variables = ['DEPTH']
for j in Data.columns[1:]:
    target_variable = j
X_train, X_test, y_train, y_test = train_test_split(Data[independent_variables], Data[target_variable], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
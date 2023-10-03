#These codes will be written in an object oriented way
import numpy as np
import pandas as pd
#Visualize missing values
import missingno as msno
import matplotlib.pyplot as plt

data = pd.read_csv('H://NISCO/codes/missed_value.csv')
Data = pd.DataFrame(data=data)
Data.drop(Data.columns[0], axis = 1 ,inplace = True)
#Numerics all data and replaces it with 'NAN' if there are no values
for i in Data.columns:
    Data[i] = pd.to_numeric(Data[i], errors = 'coerce')
a, b = Data.shape
print(Data)

#Deletes the rows including NAN - no need to sort Rows list because it made by rows and is sorted
'''Rows = []
for i in range (a):
    for j in range (b):
        DData = Data
        if np.isnan(DData.loc[i][j] ) == True:
            if i not in Rows:
                Rows.append(i)
         
for n in Rows:
   Data.drop(Data.index[n], axis = 0, inplace = True)
   print(Data)
   for i in range (len(Rows)):
       Rows[i] = Rows[i] - 1'''

#replacing NAN with zero
'''for i in range (a):
    for j in range (b):
        if np.isnan(Data.loc[i][j] ) == True:
            Data.replace(Data.loc[i][j], 0, inplace=True)'''

#Replace with mean
'''for i in range (a):
    for j in range (b):
        if np.isnan(Data.loc[i][j]) == True:
            Data[Data.columns[j]].fillna(Data[Data.columns[j]].mean(), inplace = True)
            print(Data)'''

#Replace with median
'''for i in range (a):
    for j in range (b):
        if np.isnan(Data.loc[i][j]) == True:
            Data[Data.columns[j]].fillna(Data[Data.columns[j]].median(), inplace = True)'''

#Replace with most common
'''for i in range (a):
    for j in range (b):
        if np.isnan(Data.loc[i][j]) == True:
            Data[Data.columns[j]].fillna(Data[Data.columns[j]].mode()[0], inplace = True)'''

#Replace with the previous number, forward-filling---The first row is filled using the last row's element
'''new_Data = Data.copy()
for i in range (a):
    for j in range (b):
        if np.isnan(new_Data.iloc[i, j]):
            new_Data.iloc[i, j] = new_Data.iloc[i-1, j]
print(new_Data)'''
        
#Replace with the next number, backward-filling-اگر چند تا نن پشت هم باشه تا وقتی که نن پشت سر همه و به عدد نرسیده 0 میذاره جاش
'''new_Data = Data.copy()
for j in range (b):
    if np.isnan(new_Data.iloc[a-1, j]):
        new_Data.iloc[a-1, j] = 0
for i in range (a-1):
    for j in range (b):
        if np.isnan(new_Data.iloc[i+1, j]) and np.isnan(new_Data.iloc[i, j]):
            new_Data.iloc[i, j] = 0
        elif np.isnan(new_Data.iloc[i, j]):
            new_Data.iloc[i, j] = new_Data.iloc[i+1, j]
print(new_Data)'''

#KNN-- n_neighbors and weights can change, these can be taken from user (weights="distance" or weights="uniform")
'''new_Data = Data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3, weights="distance")
a = imputer.fit_transform(new_Data)
Data_KNN = pd.DataFrame(a)
print(Data_KNN)'''

#Fill Missing Data -- linear regression -- for first and last row NAN replace with mean()
'''new_Data = Data.copy()
for j in range (b):
    if np.isnan(new_Data.iloc[a-1, j]):
        new_Data.iloc[a-1, j] = new_Data[new_Data.columns[j]].mean()
for i in range (a-1):
    for j in range (b):
        if np.isnan(new_Data.iloc[i+1, j]) and np.isnan(new_Data.iloc[i, j]):
            new_Data.iloc[i, j] = new_Data[new_Data.columns[j]].mean()
        elif np.isnan(new_Data.iloc[i, j]):
            new_Data.iloc[i, j] = (new_Data.iloc[i+1, j] + new_Data.iloc[i-1, j])/2
print(new_Data)'''

#Fill Missing Data -- polynomial regression 


#Visualize missing values
'''msno.matrix(Data)
plt.show()
msno.heatmap(Data)
plt.show()
msno.bar(Data)
plt.show()'''


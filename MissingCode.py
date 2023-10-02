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
a = Data.shape[0] #numbers of rows
b = Data.shape[1] #numbers of columns
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
            Data[Data.columns[j]].fillna(Data[Data.columns[j]].max(), inplace = True)'''


#Fill Missing Data With interpolate()--linear and Polynomial 



#Replace with the previous number-forward-filling---problem: first row 


#backward-filling





#Visualize missing values
'''msno.matrix(Data)
plt.show()
msno.heatmap(Data)
plt.show()
msno.bar(Data)
plt.show()'''

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
class main_class:
    def __init__ (self, file_Path):
        self.data = file_Path
        self.data = pd.read_csv(file_Path)
        self.data.drop(self.data.columns[0], axis = 1 ,inplace = True)
        for i in self.data.columns:
            self.data[i] = pd.to_numeric(self.data[i], errors="coerce")    #Numerics all data and replaces it with 'NAN' if there are no value
            a, b = self.data.shape
        print(self.data)        
class methods(main_class):                                                 #Inherit from the main class
    def __init__ (self, file_Path):
        main_class.__init__(self, file_Path)
    def Del_rows(self):                                                    #Deletes the rows including NAN
        a, b = self.data.shape
        Rows = []
        for i in range (a):
            for j in range (b):
                DData = self.data
                if np.isnan(DData.loc[i][j] ) == True:
                    if i not in Rows:
                        Rows.append(i)
        for n in Rows:
            self.data.drop(self.data.index[n], axis = 0, inplace = True)
            for i in range (len(Rows)):
                Rows[i] = Rows[i] - 1
        print(self.data)
    def replacement_zero(self):                                             #replacing NAN with zero
        a, b = self.data.shape   
        for i in range (a):
            for j in range (b):
                if np.isnan(self.data.loc[i][j] ) == True:
                    self.data.replace(self.data.loc[i][j], 0, inplace=True)
        print(self.data)
    def replacement_mean(self):                                             #replacing NAN with columns mean
        a, b = self.data.shape
        for i in range (a):
            for j in range (b):
                if np.isnan(self.data.loc[i][j]) == True:
                    self.data[self.data.columns[j]].fillna(self.data[self.data.columns[j]].mean(), inplace = True)
        print(self.data)
    def replacement_median(self):                                           #replacing NAN with columns median
        a, b = self.data.shape
        for i in range (a):
            for j in range (b):
                if np.isnan(self.data.loc[i][j]) == True:
                    self.data[self.data.columns[j]].fillna(self.data[self.data.columns[j]].median(), inplace = True)
        print(self.data)
    def replacement_mode(self):                                             #replacing NAN with columns mode
        a, b = self.data.shape
        for i in range (a):
            for j in range (b):
                if np.isnan(self.data.loc[i][j]) == True:
                    self.data[self.data.columns[j]].fillna(self.data[self.data.columns[j]].mode()[0], inplace = True)
        print(self.data)
    def  forward_filling(self):                                              #forward-filling---The first row is filled using the last row's element
        a, b = self.data.shape
        for i in range (a):
            for j in range (b):
                if np.isnan(self.data.iloc[i, j]):
                    self.data.iloc[i, j] = self.data.iloc[i-1, j]
        print(self.data)
    def backward_filling(self):                                              #backward-filling
        a, b = self.data.shape
        for j in range (b):
            if np.isnan(self.data.iloc[a-1, j]):
                self.data.iloc[a-1, j] = 0
            for i in range (a-1):
                for j in range (b):
                    if np.isnan(self.data.iloc[i+1, j]) and np.isnan(self.data.iloc[i, j]):
                        self.data.iloc[i, j] = 0
                    elif np.isnan(self.data.iloc[i, j]):
                        self.data.iloc[i, j] = self.data.iloc[i+1, j]
        print(self.data)
    def KNN_method(self):                                                    #KNN method
        a, b = self.data.shape
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3, weights="distance")
        imp_data = imputer.fit_transform(self.data)
        Data_KNN = pd.DataFrame(imp_data)
        print(Data_KNN)
    def lin_reg(self):                                                       #linear regression -- for first and last row NAN replace with mean()
        a, b = self.data.shape
        for j in range (b):
            if np.isnan(self.data.iloc[a-1, j]):
                self.data.iloc[a-1, j] = self.data[self.data.columns[j]].mean()
        for i in range (a-1):
            for j in range (b):
                if np.isnan(self.data.iloc[i+1, j]) and np.isnan(self.data.iloc[i, j]):
                    self.data.iloc[i, j] = self.data[self.data.columns[j]].mean()
                elif np.isnan(self.data.iloc[i, j]):
                    self.data.iloc[i, j] = (self.data.iloc[i+1, j] + self.data.iloc[i-1, j])/2
        print(self.data)
    def Visualize_miss(self):                                                #Visualize missing value
        a, b = self.data.shape
        msno.matrix(self.data)
        plt.show()
        msno.heatmap(self.data)
        plt.show()
        msno.bar(self.data)
        plt.show()
    #def polynomial regression will be add
test = methods('H://NISCO/codes/missed_value.csv')
print(test.replacement_mean())







#Dealing with missing data
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MainClass:
    def __init__ (self, path):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data.drop(self.data.columns[0], axis = 1 , inplace = True)
        #Numerics all data and replaces the missing data with NAN
        for i in self.data.columns:
            self.data[i] = pd.to_numeric(self.data[i], errors="coerce")     


class Methods(MainClass):                                                 
    def __init__ (self, path):
        MainClass.__init__(self, path)
        
    #Deletes the rows including NAN
    def delete_rows(self):
        data = self.data.copy()                                                   
        a, b = data.shape
        Rows = []
        for i in range(a):
            for j in range(b):
                Data = data
                if np.isnan(Data.loc[i][j]):
                    if i not in Rows:
                        Rows.append(i)
        for n in Rows:
            data.drop(data.index[n], axis = 0, inplace = True)
            for i in range(len(Rows)):
                Rows[i] = Rows[i] - 1
        return data

    #replacing NAN with zero
    def replacement_zero(self):                                             
        data = self.data.copy()
        a, b = data.shape   
        for i in range(a):
            for j in range(b):
                if np.isnan(data.loc[i][j]):
                    data.replace(data.loc[i][j], 0, inplace=True)
        return data

    #replacing NAN with mean
    def replacement_mean(self):                                             
        data = self.data.copy()                                                
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.loc[i][j]):
                    data[data.columns[j]].fillna(data[self.data.columns[j]].mean(), inplace = True)
        return data

    #replacing NAN with median
    def replacement_median(self):                                           
        data = self.data.copy()                                                  
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.loc[i][j]):
                    data[data.columns[j]].fillna(data[data.columns[j]].median(), inplace = True)
        return data

    #replacing NAN with mode
    def replacement_mode(self):                                            
        data = self.data.copy()                                               
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.loc[i][j]):
                    data[data.columns[j]].fillna(data[self.data.columns[j]].mode()[0], inplace = True)
        return data

    #forward-filling, The first row is filled using the last row's element
    def forward_filling(self):                                              
        data = self.data.copy()                                               
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.iloc[i, j]):
                    data.iloc[i, j] = data.iloc[i-1, j]
        return data

    #backward-filling
    def backward_filling(self):                                              
        data = self.data.copy()                                                 
        a, b = data.shape
        for j in range(b):
            if np.isnan(data.iloc[a-1, j]):
                data.iloc[a-1, j] = 0
            for i in range(a-1):
                for j in range(b):
                    if np.isnan(data.iloc[i+1, j]) and np.isnan(data.iloc[i, j]):
                        data.iloc[i, j] = 0
                    elif np.isnan(data.iloc[i, j]):
                        data.iloc[i, j] = data.iloc[i+1, j]
        return data

        #neighbors, KNN method
    def knn_method(self):                                                    
        data = self.data.copy()                                                     
        a, b = data.shape
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3, weights="distance")
        imp_data = imputer.fit_transform(data)
        Data_KNN = pd.DataFrame(imp_data)
        return Data_KNN

    #linear regression
    def linear_reg(self):                                                       
        data = self.data.copy()                                                       
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.iloc[i, j]):
                    x_values = data.dropna(subset = [data.columns[j]]).index.values.reshape(-1, 1)
                    y_values = data.dropna(subset = [data.columns[j]])[data.columns[j]].values.reshape(-1, 1)

                    model = np.polyfit(x_values.flatten(), y_values.flatten(), 1)
                    predicted_value = np.polyval(model, i)
                    data.iloc[i, j] = predicted_value
        return data
    #polynomial regression
    def polynomial_reg(self):
        data = self.data.copy()                                                         
        a, b = data.shape
        for i in range(a):
            for j in range(b):
                if np.isnan(data.iloc[i, j]):
                    x_values = data.dropna(subset = [data.columns[j]]).index.values.reshape(-1, 1)
                    y_values = data.dropna(subset = [data.columns[j]])[self.data.columns[j]].values.reshape(-1, 1)

                    #polynomial regression, degree = 2
                    poly = PolynomialFeatures(degree=2)
                    x_poly = poly.fit_transform(x_values)
            
                    model = LinearRegression()
                    model.fit(x_poly, y_values)
            
                    predicted_value = model.predict(poly.transform(np.array([[i]])))
                    data.iloc[i, j] = predicted_value[0][0]
        return data

    #Visualize missing value
    def visualize_nan(self):                                                
        data = self.data.copy()                                                         
        a, b = data.shape
        msno.matrix(data)
        plt.show()
        msno.heatmap(data)
        plt.show()
        msno.bar(data)
        plt.show()


test = Methods('../missed_value.csv')
print(test.replacement_zero())







"""
This Python code supports the following methods
to check and deal with missing data in the data preprocessing stage:
Deleting Rows with missing values, Prediction of missing values 
and filling with mean/median/mode values or filling with zero
"""

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MissingValue:
    """
    
    """
    def __init__ (self, path):
        """

        """
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data.drop(self.data.columns[0], axis = 1 , inplace = True)
        print(self.data)
        #delete unnamed columns
        self.data = self.data.iloc[:, ~self.data.columns.str.contains('Unnamed', case=False)]
        print(self.data)
class MissingValueImputer(MissingValue):   
    """
    
    """                                              
    def __init__ (self, path):
        MissingValue.__init__(self, path)
        
    #Deletes the rows including NAN
    def deleting_rows(self):
        """
        
        """
        data = self.data.copy()                                                   
        row_count, column_count = data.shape
        rows_with_nan = []
        for row_index in range(row_count):
            for column_index in range(column_count):
                Data = data
                if np.isnan(Data.loc[row_index][column_index]):
                    if row_index not in rows_with_nan:            
                        rows_with_nan.append(row_index)
        for missed_value in rows_with_nan:
            data.drop(data.index[missed_value], axis = 0, inplace = True)
            for row_index in range(len(rows_with_nan)):
                rows_with_nan[row_index] = rows_with_nan[row_index] - 1
        return data

    #replacing NAN with zero
    def filling_zero(self):  
        """

        """                                           
        data = self.data.copy()
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.loc[row_index][column_index]):
                    data.replace(data.loc[row_index][column_index], 0, inplace=True)
        return data

    #replacing NAN with mean
    def filling_mean(self): 
        """
        
        """                                            
        data = self.data.copy()                                                
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[self.data.columns[column_index]].mean(), inplace = True)
        return data

    #replacing NAN with median
    def filling_median(self):
        """
        
        """                                           
        data = self.data.copy()                                                  
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[data.columns[column_index]].median(), inplace = True)
        return data

    #replacing NAN with mode
    def filling_mode(self): 
        """
        
        """                                           
        data = self.data.copy()                                               
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[self.data.columns[column_index]].mode()[0], inplace = True)
        return data

    #forward-filling, The first row is filled using the last row's element
    def forward_filling(self):  
        """
        
        """                                            
        data = self.data.copy()                                               
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.iloc[row_index, column_index]):
                    data.iloc[row_index, column_index] = data.iloc[row_index-1, column_index]
        return data

    #backward-filling
    def backward_filling(self):
        """
        
        """                                              
        data = self.data.copy()                                                 
        row_count, column_count = data.shape
        for column_index in range(column_count):
            if np.isnan(data.iloc[row_count-1, column_index]):
                data.iloc[row_count-1, column_index] = 0
            for row_index in range(row_count-1):
                for column_index in range(column_count):
                    if (np.isnan(data.iloc[row_index+1, column_index]) and
                                                        np.isnan(data.iloc[row_index, column_index])):
                        data.iloc[row_index, column_index] = 0
                    elif np.isnan(data.iloc[row_index, column_index]):
                        data.iloc[row_index, column_index] = data.iloc[row_index+1, column_index]
        return data

        #neighbors, KNN method
    def k_nearest_neighbors_method(self, number_of_neighbors):    
        """
        
        """                                                
        self.neighbors = number_of_neighbors
        data = self.data.copy()                                                     
        knnimputer = KNNImputer(n_neighbors=number_of_neighbors, weights="distance")
        imp_data = knnimputer.fit_transform(data)
        knn_model = pd.DataFrame(imp_data)
        return knn_model

    #linear regression
    def linear_regression(self):  
        """
        
        """                                                     
        data = self.data.copy()                                                       
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.iloc[row_index, column_index]):
                    x_values = data.dropna(subset = [data.columns[column_index]]).index.values.reshape(-1, 1)
                    y_values = data.dropna(subset = [data.columns[column_index]])[data.columns[column_index]].values.reshape(-1, 1)

                    model = np.polyfit(x_values.flatten(), y_values.flatten(), 1)
                    predicted_value = np.polyval(model, row_index)
                    data.iloc[row_index, column_index] = predicted_value
        return data
    #polynomial regression
    def polynomial_regression(self, degree):
        """
        
        """
        self.degree = degree
        data = self.data.copy()                                                         
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                if np.isnan(data.iloc[row_index, column_index]):
                    x_values = data.dropna(subset = [data.columns[column_index]]).index.values.reshape(-1, 1)
                    y_values = data.dropna(subset = [data.columns[column_index]])[self.data.columns[column_index]].values.reshape(-1, 1)

                    
                    poly_feature = PolynomialFeatures(degree)
                    x_poly = poly_feature.fit_transform(x_values)
            
                    lin_reg = LinearRegression()
                    lin_reg.fit(x_poly, y_values)
            
                    predicted_value = lin_reg.predict(poly_feature.transform(np.array([[row_index]])))
                    data.iloc[row_index, column_index] = predicted_value[0][0]
        return data

    #Visualize missing value
    def Visualize_missing_value(self): 
        """
        
        """                                               
        data = self.data.copy()                                                         
        msno.matrix(data)
        plt.show()
        msno.heatmap(data)
        plt.show()
        msno.bar(data)
        plt.show()


test = MissingValueImputer('../MissingValue_data/missed_value.csv')
print(test)







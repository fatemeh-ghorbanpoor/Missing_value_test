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
    A class to receive the csv file and make it prepared for handling missing value.

    ...
    
    Attributes
    ----------
    path: str
        relative path of the file
    """
    def __init__ (self, path):
        """
        receive the relative path of the file, 
        delete unnamed columns and Inserting NAN instead of missing values
        
        Parameters
        ----------
            path: str
                relative path of the file
            Returns:
                prepared datafile for use in functions
        """
        self.path = path
        self.data = pd.read_csv(self.path)
        #Inserting NAN instead of missing values and change string values to nummeric
        for i in self.data.columns:
            self.data[i] = pd.to_numeric(self.data[i], errors="coerce")
        #delete unnamed columns
        self.data = self.data.iloc[:, ~self.data.columns.str.contains('Unnamed', case=False)]
  

class MissingValueImputer(MissingValue):   
    """
    The class for dealing with missing data, 
    inherited from the MissingValue class 
    it provides different methods for dealing with missing data.

    methods
    -------
    __init__ (path):
        inherited from the MissingValue class and receive the relative path of the file 
    deleting_rows:
        delete rows include missing value values
    filling_zero:
        replacing missing values with zero number
    filling_mean:
        replacing missing values with columns mean
    filling_median:
        replacing missing values with columns median
    filling_mode:
        replacing missing values with columns mode
    forward_filling:
        replacing missing values with the previous non-missing value
    backward_filling:
        replacing missing values with the next non-missing value
    k_nearest_neighbors_method(number_of_neighbors):
        This imputer utilizes the k-Nearest Neighbors method to replace the missing values in the datasets
        The user can give the value of the number_of_neighbors to the function
    linear_regression:
        use the other variables in the dataset to predict the missing values by linear_regression
    polynomial_regression (degree):
        use the other variables in the dataset to predict the missing values by polynomial_regression
        The user can give the value of the degree of polynomial regression to the function
    Visualize_missing_value:
        Heatmap, matrix and bar charts are used for visually representing the occurrence of missing values across different variables
    """    

    def __init__ (self, path):
        """
        receive the relative path of the file, 
        delete unnamed columns and Inserting NAN instead of missing values
        
        Parameters
        ----------
            path: str
                relative path of the file
        """
        MissingValue.__init__(self, path)
        
    def deleting_rows(self):
        """
        removes rows containing missing values
        Returns:
            A dataset without missing values
        """
        data = self.data.copy()                                                   
        row_count, column_count = data.shape
        #A list to put rows containing missing values ​​in
        rows_with_nan = []
        for row_index in range(row_count):
            for column_index in range(column_count):
                #makes it able to correctly recognize the row and column index ​​in each loop
                Data = data
                #Detection of missing values and inserting row_index into the list
                if np.isnan(Data.loc[row_index][column_index]):
                    if row_index not in rows_with_nan:            
                        rows_with_nan.append(row_index)
        #removes rows containing missing values
        for missed_value in rows_with_nan:
            data.drop(data.index[missed_value], axis = 0, inplace = True)
            for row_index in range(len(rows_with_nan)):
                rows_with_nan[row_index] = rows_with_nan[row_index] - 1
        return data

    def filling_zero(self):  
        """
        replacing missing values with zero number
        Returns:
            A dataset without missing values
        """         
        #                                 
        data = self.data.copy()
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values and replacing missing values with zero number
                if np.isnan(data.loc[row_index][column_index]):
                    data.replace(data.loc[row_index][column_index], 0, inplace=True)
        return data

    def filling_mean(self): 
        """
        replacing missing values with column mean
        Returns:
            A dataset without missing values
        """                                              
        data = self.data.copy()                                                
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values and replacing missing values with column mean
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[self.data.columns[column_index]].mean(), inplace = True)
        return data

    def filling_median(self):
        """
        replacing missing values with column median
        Returns:
            A dataset without missing values
        """                                             
        data = self.data.copy()                                                  
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values and replacing missing values with column median
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[data.columns[column_index]].median(), inplace = True)
        return data

    def filling_mode(self): 
        """
        replacing missing values with column mode
        Returns:
            A dataset without missing values
        """                                            
        data = self.data.copy()                                               
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values and replacing missing values with column mode
                if np.isnan(data.loc[row_index][column_index]):
                    data[data.columns[column_index]].fillna(data[self.data.columns[column_index]].mode()[0], inplace = True)
        return data

    def forward_filling(self):  
        """
        replacing missing values with the previous non-missing value
        Returns:
            A dataset without missing values
        """                                                
        data = self.data.copy()                                               
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values and replacing missing value by forward filling method
                if np.isnan(data.iloc[row_index, column_index]):
                    data.iloc[row_index, column_index] = data.iloc[row_index-1, column_index]
        return data

    def backward_filling(self):
        """
        replacing missing values with the next non-missing value
        Returns:
            A dataset without missing values
        """                                                
        data = self.data.copy()                                                 
        row_count, column_count = data.shape
        #it only checks the last row and replaces the missing values ​​with zero
        for column_index in range(column_count):
            if np.isnan(data.iloc[row_count-1, column_index]):
                data.iloc[row_count-1, column_index] = 0
        #it only checks other rows and If the next data of missing value is also missing value. It is replaced by zero 
            for row_index in range(row_count-1):
                for column_index in range(column_count):
                    if np.isnan(data.iloc[row_index+1, column_index]) and np.isnan(data.iloc[row_index, column_index]):
                        data.iloc[row_index, column_index] = 0
                    #replaces the missing values ​by backward filling method
                    elif np.isnan(data.iloc[row_index, column_index]):
                        data.iloc[row_index, column_index] = data.iloc[row_index+1, column_index]
        return data

    def k_nearest_neighbors_method(self, number_of_neighbors):    
        """
        using the k-Nearest Neighbors method to replace the missing values in the datasets
        Parameters
        ----------
        number_of_neighbors: int
            The number of missing data neighbors 
        Returns:
            A dataset without missing values
        """                                                  
        self.neighbors = number_of_neighbors
        data = self.data.copy()   
        #It takes the number of neighbors                                             
        knnimputer = KNNImputer(n_neighbors=number_of_neighbors, weights="distance")
        #fits the data within knn method
        imp_data = knnimputer.fit_transform(data)
        knn_model = pd.DataFrame(imp_data)
        return knn_model

    def linear_regression(self):  
        """
        using the linear regression method to replace the missing values in the datasets
        Returns:
            A dataset without missing values
        """                                                      
        data = self.data.copy()                                                       
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values
                if np.isnan(data.iloc[row_index, column_index]):
                    #selection of available data for linear regression
                    x_values = data.dropna(subset = [data.columns[column_index]]).index.values.reshape(-1, 1)
                    y_values = data.dropna(subset = [data.columns[column_index]])[data.columns[column_index]].values.reshape(-1, 1)
                    #fits the data within a linear regression
                    model = np.polyfit(x_values.flatten(), y_values.flatten(), 1)
                    predicted_value = np.polyval(model, row_index)
                    data.iloc[row_index, column_index] = predicted_value
        return data

    def polynomial_regression(self, degree):
        """
        using the polynomial regression method to replace the missing values in the datasets
        Parameters
        ----------
        degree: int
            value of the degree of polynomial regression
        Returns:
            A dataset without missing values
        """  
        self.degree = degree
        data = self.data.copy()                                                         
        row_count, column_count = data.shape
        for row_index in range(row_count):
            for column_index in range(column_count):
                #Detection of missing values
                if np.isnan(data.iloc[row_index, column_index]):
                    #Selection of available data for polynomial regression
                    x_values = data.dropna(subset = [data.columns[column_index]]).index.values.reshape(-1, 1)
                    y_values = (data.dropna(subset = [data.columns[column_index]])[self.data.columns[column_index]].values.reshape(-1, 1))

                    #fits the data within a Polynomial regression
                    poly_feature = PolynomialFeatures(degree)
                    x_poly = poly_feature.fit_transform(x_values)
            
                    lin_reg = LinearRegression()
                    lin_reg.fit(x_poly, y_values)
            
                    predicted_value = lin_reg.predict(poly_feature.transform(np.array([[row_index]])))
                    data.iloc[row_index, column_index] = predicted_value[0][0]
        return data

    def Visualize_missing_value(self): 
        """
        using matrix and bar charts for visually representing the occurrence of missing values across different variables
        Returns:
            A dataset without missing values
        """                                                
        data = self.data.copy()                                                         
        msno.matrix(data)
        plt.show()
        msno.heatmap(data)
        plt.show()
        msno.bar(data)
        plt.show()


test = MissingValueImputer('../MissingValue_data/missed_value.csv')
print(test.k_nearest_neighbors_method(5))








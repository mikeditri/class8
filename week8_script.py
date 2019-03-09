#!/usr/bin/env python

import sklearn.datasets
from sklearn.datasets import load_wine
wine_data = load_wine()


##to get list of available things to explore print(dir(wine_data))
#need to list properties of each part of the data set and list what they mean
#to explore the data do: wine_data['DESCR']
# 'DESCR'>> Descriptions
# 'data' = actual data
# 'feature_names'= names of the columns
# 'target' = list of all the classes
# 'target_names' = list of all the classes by name
#TODO
#1 first, visualize your data from loading it this way
# this can actually include converting to pandas, and running your script from before
#2 train & apply a KNN-classifier or KNN-regressor on your dataset
#3 print the output "score" or "performance" of the classifier/regressor
# if you did 2 and 3 right your performance will not be perfect

import pandas as pd

wine_df = pd.DataFrame(data=wine_data[1], columns = wine_data[4])
print(wine_df)


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
import numpy as np
import pandas as pd

#print(wine_data['target'])

wine_df = pd.DataFrame(data=np.c_[wine_data['data'],wine_data['target']],columns=wine_data['feature_names']+['Class'])
wine_df = wine_df.rename(columns = {'od280/od315_of_diluted_wines':'od280_od315_of_diluted_wines'})

#print(wine_df.columns)

print()
print("Mean of Each Variable")
print(np.mean(wine_df))
#Space
print()
print("Standard deviation of Each Variable")
print(np.std(wine_df))

#4. Look at it
#4.1 Plot values on histograms
#    import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
#4.1.1  for each column:
#           get values
#           plot histogram
#           save to file as column_name.png
x_label = (list(wine_df.columns.values))

for i in wine_df.columns:
    idx = wine_df.columns.get_loc(i)
    fig = plt.hist(wine_df.iloc[:,idx])
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.savefig(i+'_hist.png')
    plt.clf()


#n = len(data.columns)
		
#for i in range(n):
#	for j in range(i):
#		plt.scatter(data.iloc[:,i],data.iloc[:,j])
#		i_name = data.columns[i]
#		j_name = data.columns[j]
#		plt.xlabel(i_name)
#		plt.ylabel(j_name)
#		plt.savefig(i_name+'_vs_'+j_name+'scatter_pairs.png')
#		plt.clf()
		
#4.3 Plot pair-wise correlation matrix as a heatmap
correlations = wine_df.corr()
names = wine_df.columns
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,rotation = 90)
ax.set_yticklabels(names, rotation = 0)
plt.tight_layout()
plt.savefig('correlation_matrix.png')

import seaborn as sns
from tqdm import tqdm

# Convert columns into categories to fix next graphs

wine_df.iloc[:,13] = wine_df.iloc[:,13].replace(0.0,'Class_0')
wine_df.iloc[:,13] = wine_df.iloc[:,13].replace(1.0,'Class_1')
wine_df.iloc[:,13] = wine_df.iloc[:,13].replace(2.0,'Class_2')

n = len(wine_df.columns)


for i in tqdm(range(n)):
	for j in range(i):
		for k in range(j):
			i_name = wine_df.columns[i]
			j_name = wine_df.columns[j]
			k_name = wine_df.columns[k]
			sns.relplot(x=i_name,y=j_name,data=wine_df, hue= 'Class',palette=['r','g','b'],size=k_name,sizes=(15, 200))
			plt.xlabel(i_name)
			plt.ylabel(j_name)
			plt.savefig(i_name+'_vs_'+j_name+'_size_'+k_name+'_scatter_pairs.png')
			plt.clf()


#data["Class"] = data["Class"].astype(str)
#sns.relplot(x='Proline',y='OD280 OD315 of diluted wines',data=data,legend='brief',size='Flavanoids',sizes=(15, 200),hue="Class",palette=["r","g","b"])
#plt.show()

############## credit to https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#preprocessing
scaler = preprocessing.StandardScaler()
Scaled_data = scaler.fit_transform(wine_data.data)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(Scaled_data[:,11:13], wine_data.target, test_size=0.3) # 70% training and 30% test

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#used this to determine K value

#K_accuracy_val = []
#for kx in range(120):
#	kx = kx+1
#	#Create KNN Classifier
#	knn = KNeighborsClassifier(n_neighbors=kx, weights='distance')
#	#Train the model using the training sets
#	knn.fit(X_train, y_train)
#	#Predict the response for test dataset
#	y_pred = knn.predict(X_test)
#	K_accuracy = metrics.accuracy_score(y_test, y_pred)
#	K_accuracy_val.append(K_accuracy) #store values
#	# Model Accuracy, how often is the classifier correct?
#	#print("Accuracy for k:", kx, ":",K_accuracy)

#curve = pd.DataFrame(K_accuracy_val) #elbow curve 
#curve.plot()
#plt.show()

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')

# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

knn.fit(X_train,y_train)

# Plotting decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap=cmap_bold)
plt.scatter(X_test[:,0],X_test[:,1], c=y_test, cmap = cmap_light)
plt.xlabel(wine_data.feature_names[11])
plt.ylabel(wine_data.feature_names[12])
y_pred = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
plt.title('Accuracy:'+ str(accuracy))
plt.legend(y_train,scatterpoints = 1)
plt.savefig('KNN_pairs_'+'od280_od315_of_diluted_wines'+'_&_'+wine_data.feature_names[12]+'.png')
plt.clf()



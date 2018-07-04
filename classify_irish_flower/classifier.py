from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

#VARIABLES 
iris = datasets.load_iris()

#FUNCTIONS
def shuffler( d, t):
	aux = np.c_[d.reshape(len(d), -1), t.reshape(len(t), -1)]
	aux = shuffle(aux)
	aux_d = aux[:, :d.size//len(d)].reshape(d.shape)
	aux_t = aux[:, d.size//len(d):].reshape(t.shape)
	
	return aux_d, aux_t



print("FEATURES:") 
print(iris.feature_names)
print("TARGETS [0,1,2]:")
print(iris.target_names)
print("#################")

#Shuffle the data 
(iris.data, iris.target)=shuffler(iris.data, iris.target)

#Select test data  and training data
train_data, train_target, test_data, test_target = (iris.data[:int(len(iris.data)*0.7)] 
						,iris.target[:int(len(iris.target)*0.7)]
						,iris.data[int(len(iris.data)*0.7):]
						,iris.target[int(len(iris.target)*0.7):])	     	 

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import PolynomialFeatures
for i in range(10):
	poly = PolynomialFeatures(degree = i)
	train_p = poly.fit_transform(train_data)	
	test_p = poly.fit_transform(test_data)
	gnb = GaussianNB()
	gnb.fit(train_p, train_target)
	print("Accuracy with naive dim %d bayes: %f" % (i, accuracy_score(gnb.predict(test_p),test_target)))  

#Plotting
"""
c = [ "b", "r", "g"]
for f1, f2, col in zip(train_data[:,3], train_data[:,1], train_target):
	plt.scatter(f1, f2, color = c[int(col)], label="data") 
plt.show()
"""

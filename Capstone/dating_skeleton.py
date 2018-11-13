import pandas as pd
import numpy as np
from analysisModules import *
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, r2_score
from itertools import combinations 
import time
import seaborn

data = pd.read_csv("profiles_processed.csv")

def knnGraph(features, label, stepRange):
    feature_data = features
    x = feature_data.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    train_feature, test_feature, train_label, test_label = train_test_split(feature_data, label, train_size=0.8, test_size=0.2, random_state=5)
    k_values = stepRange
    score_values = []
    startTime = time.time()
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_feature, train_label)
        score = classifier.score(test_feature, test_label)
        print('knn with k=', k, '\tscore=', score)
        score_values.append(score)
    return k_values, score_values, time.time() - startTime

def knnRegr(features, label, stepRange):
    feature_data = features
    x = feature_data.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    train_feature, test_feature, train_label, test_label = train_test_split(feature_data, label, train_size=0.8, test_size=0.2, random_state=5)
    k_values = stepRange
    score_values = []
    startTime = time.time()
    for k in k_values:
        classifier = KNeighborsRegressor(n_neighbors=k)
        classifier.fit(train_feature, train_label)
        score = classifier.score(test_feature, test_label)
        print('knn regressor with k=', k, '\tscore=', score)
        score_values.append(score)
    return k_values, score_values, time.time() - startTime


def SVCGraph(features, label):
    feature_data = features
    x = feature_data.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    train_feature, test_feature, train_label, test_label = train_test_split(feature_data, label, train_size=0.8, test_size=0.2, random_state=5)
    kernels = ['linear', 'rbf', 'sigmoid']
    scores = []
    startTime = time.time()
    for k in kernels:
        classifier = SVC(kernel=k)
        classifier.fit(train_feature, train_label)
        score = classifier.score(test_feature, test_label)
        print('Kernel: ', k, '\tScore: ', score)
        scores.append(score)
    return kernels, scores, time.time() - startTime

def NaiveBayes(features, label):
    feature_data = features
    x = feature_data.values
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    train_feature, test_feature, train_label, test_label = train_test_split(feature_data, label, train_size=0.8, test_size=0.2, random_state=5)
    classifier= MultinomialNB()
    classifier.fit(train_feature, train_label)
    score = classifier.score(test_feature, test_label)
    return score

def plot(x, y, title, xlabel, ylabel, filename):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)


def linearTest(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
	lm = LinearRegression()
	model = lm.fit(x_train, y_train)
	y_predict = lm.predict(x_test)
	return lm.score(x_train, y_train), lm.score(x_test, y_test)

#k, s, t = knnGraph(data[['education_level', 'income']].fillna(0), data['sex_code'], range(100, 200, 5))
#plot(k, s, 'KNN: Determine sex by education level and income', 'k value', 'score', './analysis/knn_sex.png')

""" 
k, s, t = knnGraph(data[['likes_pets', 'income', 'sex_code' ,'education_level']].fillna(0), data['likes_children'], range(10, 200, 10))
print(np.amax(s), t/len(s))
plot(k, s, 'KNN: Determine If likes children or not by pets, income, education and sex', 'k value', 'score', './analysis/knn_children.png')

k, s, t = SVCGraph(data[['likes_pets', 'income', 'sex_code' ,'education_level']].fillna(0), data['likes_children'])
print(np.amax(s), t/len(s))
plot(k, s, 'SVC: Determine If likes children or not by pets, income and sex', 'Kernel', 'score', './analysis/svc_children.png')
"""

removeOutlierSets = data
rs = removeOutlierSets
rs = rs[(rs.income<400000) & (rs.income>0) & (rs.essay_len<50000) ].fillna(0)

income = rs['income']

all_features = rs[['age', 'education_level', 'essay_len', 'smokes_code', 'drinks_code', 'sex_code', 'likes_pets', 'likes_children', 'height']]


fsize = len(all_features)
highest_train = ([], (0,0))
highest_test = ([], (0,0))
for i in range(fsize):
    comb = combinations(all_features, i+1)
    for c in comb:
      scores = linearTest(all_features[list(c)], income)
      if scores[0] > highest_train[1][0]:
        highest_train = (c, scores)
        print(c, scores)
      if scores[1] > highest_test[1][1]:
        highest_test = (c, scores)
        print(c, scores)
  
print("highest scores:")
print(highest_train)
print(highest_test)

#print(np.amax(s), t/len(s))
#plot(k, s, 'KNN Regressor: Determine Income with essay length', 'k value', 'score', './analysis/knnregr_income2.png')


# data.education = data.education.fillna("NA")
# seaborn.pairplot(feature_data)
#plt.show()

# print(all_data.columns)
#histAllColumns(all_data, ['age', 'income', 'offspring', 'orientation', 'pets', 'speaks', 'status', 'drinks_code', 'smokes_code', 'drugs_code', 'essay_len'], './analysis')





'''
feature_data = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
zodiac = all_data['sign'].apply(lambda x: 'none' if x==0 else x.split()[0])

train_feature, test_feature, train_zodiac, test_zodiac = train_test_split(feature_data, zodiac, train_size=0.8, test_size=0.2)

k_values = range(1, 200, 1)
score_values = []
for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_feature, train_zodiac)
    score = classifier.score(test_feature, test_zodiac)
    print(score)
    score_values.append(score)

plt.plot(k_values, score_values)
plt.show()
'''

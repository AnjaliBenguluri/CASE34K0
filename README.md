import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('drowsy.csv')
print(dataset)

features = pd.read_csv('drowsy.csv',usecols=[0,1,2,3])
labels = pd.read_csv('drowsy.csv', usecols = [4])

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features,labels)
print(knn.predict([[0.4,7,0,18]]))
print(knn.score(features,labels))

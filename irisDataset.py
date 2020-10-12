from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# Load iris datasets
dataset = datasets.load_iris()

#fit a CART model to data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)

#make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

#Summarize the fit
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

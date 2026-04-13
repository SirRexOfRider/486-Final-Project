from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as dtc
#from sklearn.tree import decisiontreeregression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier as rfc


#Import the iris data as a data frame
iris = load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tree = dtc(max_depth=2, random_state=42)
tree.fit(X_train, y_train)

#This is gonna be a column
y_pred = tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"We got an accruacy of {acc:,.2f}")

plt.figure(figsize=(10,12))
plot_tree(tree, filled=True, fontsize=12, feature_names = iris.feature_names, class_names = iris.target_names)
plt.show()

#Add data, and then into one record
c = tree.predict([[2.41, 1.25]])

print(f"The prediction for the individual record is {iris.target_names[c]}")

forest = rfc(n_estimators=200, random_state = 13)

forest.fit(X_train, y_train)

y_pred_forest = forest.predict(X_test)

acc_forest = accuracy_score(y_test, y_pred_forest)
print(f"The accuray of the forest is {acc_forest}")

c = forest.predict([[2.41, 1.25]])
print(f"The prediction for the individual record is {iris.target_names[c]}")
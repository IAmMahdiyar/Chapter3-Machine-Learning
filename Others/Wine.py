from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

wine_data = load_wine()

print("Target Names:", wine_data.target_names)

X = wine_data.data
Y = wine_data.target

n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()

print("Sample Count: ", "Class0=", n_class0, "Class1=", n_class1, "Class2=", n_class2)

x_train, x_test, y_train, y_test = train_test_split(X, Y)

svc = SVC(kernel='linear')

svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

pred = svc.predict(x_test)

print(classification_report(y_test, pred))